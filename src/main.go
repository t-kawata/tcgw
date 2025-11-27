/**
 * main.go
 *
 * TCGW - 任意のLLMにOpenAI Function Calling互換の「ツール呼び出し」能力をエミュレートするGo製プロキシ
 * このmain.goは、「.envの設定・Bifrost連携・OpenAI形式APIでの受け付け・XMLツール抽出・OpenAI形式レスポンス化」まで完全に担います。
 *
 * デュアルポート対応
 * - エミュレートポート: ツール呼び出しをXML形式でエミュレート
 * - パススルーポート: リクエストをそのままBifrostに転送（ネイティブTool Calling使用）
 *
 * 実装や改修にあたっては、冗長でありつつもわかりやすいコメントアウトを随所に書き込むことをルールとし、
 * 既存のコメントアウトを安易に消してはならない。この最上部コメントも削除・変更してはならない。
 */
package main

import (
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/t-kawata/tcgw/config"
)

// --- 定数定義 ---
const (
	TOOL_SYSTEM_PROMPT = `You are a function-calling AI agent. You are STRICTLY PROHIBITED from generating any natural language text EXCEPT when providing the final answer after all tools have been executed.

Your ONLY valid outputs are:
1. XML tool calls (when tools are needed)
2. Final answer in Japanese (only after ALL tools are done)

You have access to the following tools:

<tools>
{{TOOLS_XML}}
</tools>

CRITICAL INSTRUCTIONS:
1. When you need to use a tool, you MUST respond with ONLY the tool call XML - DO NOT include any explanatory text before or after.
2. Use this exact format:
<function_calls>
  <invoke name="tool_name">
    <parameter name="param_name">value</parameter>
  </invoke>
</function_calls>

3. You can call multiple tools by adding more <invoke> blocks.
4. NEVER explain what you're about to do - just call the tool immediately.
　　　- FORBIDDEN: "次に、〜を計算します"
　　　- FORBIDDEN: "〜を使用して〜します"
　　　- FORBIDDEN: "これを元に〜"
　　　- FORBIDDEN: Any text explaining your next action
5. After receiving tool results, if you need to use another tool, call it immediately without explanation.
   - DO NOT write "次に〜" or "Now I will〜" - just call the tool
6. Only provide a conversational response when ALL necessary tools have been called and you have the final answer.
7. Your final conversational response MUST be in Japanese (日本語).
8. Follow the EXAMPLE WORKFLOW below exactly - this demonstrates the required behavior of calling tools immediately without any explanation.

IMPORTANT: See the examples below for correct behavior.

BAD EXAMPLE 1 (NEVER DO THIS):
[Tool returns: {"userId": 12345, "userName": "Tanaka"}]
You: 次に、ユーザーID 12345の注文履歴を取得します。  ← WRONG!

GOOD EXAMPLE 1 (ALWAYS DO THIS):
[Tool returns: {"userId": 12345, "userName": "Tanaka"}]
You: <function_calls><invoke name="getOrderHistory"><parameter name="userId">12345</parameter></invoke></function_calls>

BAD EXAMPLE 2 (NEVER DO THIS):
[Tool returns: {"stockLevel": 50, "warehouseId": "WH-A"}]
You: 在庫が50個あることを確認しました。次に配送スケジュールを作成します。  ← WRONG!

GOOD EXAMPLE 2 (ALWAYS DO THIS):
[Tool returns: {"stockLevel": 50, "warehouseId": "WH-A"}]
You: <function_calls><invoke name="createShipmentSchedule"><parameter name="warehouseId">WH-A</parameter><parameter name="quantity">50</parameter></invoke></function_calls>

GOOD EXAMPLE WORKFLOW:
User: "明日の東京の天気を確認して、晴れならフライトを予約して確認メールを送って"

You: <function_calls><invoke name="checkWeather"><parameter name="location">Tokyo</parameter><parameter name="date">tomorrow</parameter></invoke></function_calls>

[Tool returns: {"condition": "sunny", "temperature": 25}]

You: <function_calls><invoke name="bookFlight"><parameter name="destination">Tokyo</parameter><parameter name="date">tomorrow</parameter></invoke></function_calls>

[Tool returns: {"bookingId": "FL12345", "status": "confirmed"}]

You: <function_calls><invoke name="sendEmail"><parameter name="subject">Flight Confirmation</parameter><parameter name="body">Your flight FL12345 to Tokyo is confirmed for tomorrow</parameter></invoke></function_calls>

[Tool returns: {"status": "sent", "messageId": "MSG789"}]

You: 明日の東京の天気は晴れ（気温25度）です。フライトFL12345の予約が完了し、確認メールを送信しました。

Always use the exact tool names and parameter names as specified.`

	FUNCTION_CALLS_PATTERN = `<function_calls>([\s\S]*?)</function_calls>`
	INVOKE_PATTERN         = `<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>`
	PARAMETER_PATTERN      = `<parameter\s+name="([^"]+)">([\s\S]*?)</parameter>`
	JSON_PATTERN           = `\{[^{}]*"tool_calls"[^{}]*\}`
	MARKDOWN_JSON_PATTERN  = `(?s)` + "`" + `(?:json)?\s*([^` + "`" + `]+)` + "`" + `(?:` + "`" + `|$)`
)

// --- 正規表現 (グローバルコンパイル) ---
// パフォーマンス向上のため、正規表現を起動時に一度だけコンパイルします
var (
	reFunctionCalls = regexp.MustCompile(FUNCTION_CALLS_PATTERN)
	reInvoke        = regexp.MustCompile(INVOKE_PATTERN)
	reParameter     = regexp.MustCompile(PARAMETER_PATTERN)
	reJSON          = regexp.MustCompile(JSON_PATTERN)
	reMarkdownJSON  = regexp.MustCompile(MARKDOWN_JSON_PATTERN)
)

// --- グローバル変数 (設定) ---
var bifrostURL string
var emulatePort string     // エミュレートモード用ポート
var passthroughPort string // パススルーモード用ポート
var debugMode bool
var requestTimeout int64
var bifrostApiKey string

// --- 型定義 (リクエスト) ---

// Message はチャット会話の1メッセージを表す
type Message struct {
	Role       string     `json:"role"`                   // "system", "user", "assistant", "tool"
	Content    any        `json:"content,omitempty"`      // string or []ContentPart (マルチモーダル対応)
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // assistantメッセージでのツール呼び出し
	ToolCallID string     `json:"tool_call_id,omitempty"` // toolメッセージで必須
	Name       string     `json:"name,omitempty"`         // tool/functionメッセージで使用
	Refusal    *string    `json:"refusal,omitempty"`      // assistantが拒否した場合（レスポンスのみ）
}

// ContentPart はマルチモーダルコンテンツの一部（テキスト/画像など）
type ContentPart struct {
	Type     string    `json:"type"`                // "text", "image_url"
	Text     string    `json:"text,omitempty"`      // type="text"の場合
	ImageURL *ImageURL `json:"image_url,omitempty"` // type="image_url"の場合
}

// ImageURL は画像URLと詳細度を指定
type ImageURL struct {
	URL    string `json:"url"`              // 画像URL（https:// or data:image/...）
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// FunctionDef は関数定義
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"` // 推奨だがオプショナル
	Parameters  map[string]any `json:"parameters"`            // JSON Schema
	Strict      *bool          `json:"strict,omitempty"`      // Structured Outputs用
}

// Tool はツール定義（現在はfunctionのみ）
type Tool struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// ToolChoice はツール選択の動作を指定
type ToolChoiceObject struct {
	Type     string                     `json:"type"` // "function"
	Function ToolChoiceFunctionSelector `json:"function"`
}

type ToolChoiceFunctionSelector struct {
	Name string `json:"name"` // 強制するツール名
}

// ResponseFormat はレスポンス形式を指定（JSON mode用）
type ResponseFormat struct {
	Type       string                `json:"type"`                  // "text", "json_object", "json_schema"
	JSONSchema *ResponseFormatSchema `json:"json_schema,omitempty"` // type="json_schema"の場合
}

type ResponseFormatSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      *bool          `json:"strict,omitempty"`
}

// StreamOptions はストリーミングオプション
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"` // ストリーム終了時にusageを含めるか
}

// ChatCompletionRequest はチャット補完リクエスト
type ChatCompletionRequest struct {
	// 必須パラメータ
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`

	// ツール関連
	Tools             []Tool `json:"tools,omitempty"`
	ToolChoice        any    `json:"tool_choice,omitempty"`         // "none", "auto", "required", or ToolChoiceObject
	ParallelToolCalls *bool  `json:"parallel_tool_calls,omitempty"` // デフォルトtrue

	// サンプリングパラメータ
	Temperature      *float32 `json:"temperature,omitempty"`       // 0.0 ~ 2.0, デフォルト1.0
	TopP             *float32 `json:"top_p,omitempty"`             // 0.0 ~ 1.0, デフォルト1.0
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"` // -2.0 ~ 2.0, デフォルト0
	PresencePenalty  *float32 `json:"presence_penalty,omitempty"`  // -2.0 ~ 2.0, デフォルト0

	// 生成制御
	MaxTokens           *int   `json:"max_tokens,omitempty"`            // 旧名: max_completion_tokens
	MaxCompletionTokens *int   `json:"max_completion_tokens,omitempty"` // 新名（推奨）
	N                   *int   `json:"n,omitempty"`                     // 生成する選択肢数、デフォルト1
	Stop                any    `json:"stop,omitempty"`                  // string or []string
	Seed                *int64 `json:"seed,omitempty"`                  // 再現性用

	// ストリーミング
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`

	// ログ確率
	Logprobs    *bool `json:"logprobs,omitempty"`     // デフォルトfalse
	TopLogprobs *int  `json:"top_logprobs,omitempty"` // 0-20

	// フォーマット
	ResponseFormat any `json:"response_format,omitempty"` // ResponseFormat or map

	// その他
	User            string         `json:"user,omitempty"`             // エンドユーザー識別子（abuse検知用）
	Metadata        map[string]any `json:"metadata,omitempty"`         // カスタムメタデータ
	Store           *bool          `json:"store,omitempty"`            // 保存するか（model distillation用）
	ReasoningEffort string         `json:"reasoning_effort,omitempty"` // o1モデル用: "low", "medium", "high"

	// 音声関連（将来対応）
	Modalities []string     `json:"modalities,omitempty"` // ["text", "audio"]
	Audio      *AudioParams `json:"audio,omitempty"`

	// 予測関連（将来対応）
	Prediction *PredictionParams `json:"prediction,omitempty"`
}

// AudioParams は音声出力パラメータ
type AudioParams struct {
	Voice  string `json:"voice"`  // "alloy", "echo", "fable", "onyx", "nova", "shimmer"
	Format string `json:"format"` // "wav", "mp3", "flac", "opus", "pcm16"
}

// PredictionParams は予測補完パラメータ
type PredictionParams struct {
	Type    string    `json:"type"`    // "content"
	Content []Message `json:"content"` // 予測するコンテンツ
}

// --- 型定義 (レスポンス) ---

// ToolCallFunction はツール呼び出しの関数情報
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON文字列
}

// ToolCall はツール呼び出し情報
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

// ResponseMessage はレスポンスメッセージ
type ResponseMessage struct {
	Role      string     `json:"role"` // "assistant"
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Refusal   *string    `json:"refusal,omitempty"` // モデルが拒否した場合
	Audio     *Audio     `json:"audio,omitempty"`   // 音声出力
}

// Audio は音声レスポンス情報
type Audio struct {
	ID         string `json:"id"`
	ExpiresAt  int64  `json:"expires_at"` // Unix timestamp
	Data       string `json:"data"`       // base64エンコードされた音声データ
	Transcript string `json:"transcript"` // 音声の転写テキスト
}

// Logprobs はログ確率情報
type Logprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
	Refusal []TokenLogprob `json:"refusal,omitempty"`
}

// TokenLogprob は個別トークンのログ確率情報
type TokenLogprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes,omitempty"` // UTF-8バイト表現
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// TopLogprob は上位トークンのログ確率
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// Choice は生成された選択肢
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"` // "stop", "length", "tool_calls", "content_filter", "function_call"
	Logprobs     *Logprobs       `json:"logprobs,omitempty"`
}

// Usage はトークン使用量
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails はプロンプトトークンの詳細
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"` // キャッシュされたトークン数
	AudioTokens  int `json:"audio_tokens,omitempty"`  // 音声入力トークン数
}

// CompletionTokensDetails は補完トークンの詳細
type CompletionTokensDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`           // 推論トークン数（o1モデル）
	AudioTokens              int `json:"audio_tokens,omitempty"`               // 音声出力トークン数
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"` // 受け入れられた予測トークン数
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"` // 拒否された予測トークン数
}

// ChatCompletionResponse はチャット補完レスポンス
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`  // "chat.completion"
	Created           int64    `json:"created"` // Unix timestamp
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"` // バックエンド構成の識別子
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
	ServiceTier       string   `json:"service_tier,omitempty"` // "scale", "default"
}

// ErrorResponse はエラーレスポンス
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail はエラー詳細
type ErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param,omitempty"`
	Code    *string `json:"code,omitempty"`
}

// --- 設定初期化 ---
func initConfig() {
	_ = godotenv.Load() // .env ファイルを読み込み

	bifrostURL = os.Getenv("BIFROST_URL")
	if bifrostURL == "" {
		bifrostURL = "http://0.0.0.0:7766"
	}
	if !strings.HasPrefix(bifrostURL, "http://") && !strings.HasPrefix(bifrostURL, "https://") {
		fmt.Fprintf(os.Stderr, "❌ BIFROST_URL must start with http:// or https://\n")
		os.Exit(1)
	}
	_, err := url.Parse(bifrostURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Invalid BIFROST_URL: %v\n", err)
		os.Exit(1)
	}

	// エミュレートポート設定
	emulatePortStr := os.Getenv("EMULATE_PORT")
	if emulatePortStr == "" {
		emulatePortStr = "3000"
	}
	port, err := strconv.Atoi(emulatePortStr)
	if err != nil || port < 1 || port > 65535 {
		fmt.Fprintf(os.Stderr, "❌ EMULATE_PORT must be a number between 1 and 65535\n")
		os.Exit(1)
	}
	emulatePort = ":" + emulatePortStr

	// パススルーポート設定
	passthroughPortStr := os.Getenv("PASSTHROUGH_PORT")
	if passthroughPortStr == "" {
		passthroughPortStr = "3001"
	}
	port, err = strconv.Atoi(passthroughPortStr)
	if err != nil || port < 1 || port > 65535 {
		fmt.Fprintf(os.Stderr, "❌ PASSTHROUGH_PORT must be a number between 1 and 65535\n")
		os.Exit(1)
	}
	passthroughPort = ":" + passthroughPortStr

	// ポート重複チェック
	if emulatePort == passthroughPort {
		fmt.Fprintf(os.Stderr, "❌ EMULATE_PORT and PASSTHROUGH_PORT must be different\n")
		os.Exit(1)
	}

	timeoutStr := os.Getenv("REQUEST_TIMEOUT")
	if timeoutStr == "" {
		timeoutStr = "120000"
	}
	timeout, err := strconv.ParseInt(timeoutStr, 10, 64)
	if err != nil || timeout < 5000 || timeout > 600000 {
		fmt.Fprintf(os.Stderr, "❌ REQUEST_TIMEOUT must be between 5000 and 600000 milliseconds\n")
		os.Exit(1)
	}
	requestTimeout = timeout

	debugStr := os.Getenv("DEBUG_MODE")
	debugMode = strings.ToLower(debugStr) == "true"
	bifrostApiKey = os.Getenv("BIFROST_API_KEY")

	fmt.Println("🌉 TCGW Proxy Server (Dual-Port Mode)")
	if debugMode {
		fmt.Printf("[TCGW] Server Configuration\n  Emulate Port: %s\n  Passthrough Port: %s\n  Bifrost URL: %s\n  Debug Mode: true\n  Request Timeout: %dms\n",
			strings.TrimPrefix(emulatePort, ":"),
			strings.TrimPrefix(passthroughPort, ":"),
			bifrostURL,
			requestTimeout)
	}
	fmt.Println("[TCGW] Server Starting")
	fmt.Printf("  TC Emulate Mode:        0.0.0.0%s (Tool Calling Emulation)\n", emulatePort)
	fmt.Printf("  NO-TC Passthrough Mode: 0.0.0.0%s (Native Tool Calling)\n", passthroughPort)
	fmt.Printf("  BIFROST:                %s\n", bifrostURL)
}

// --- ヘルパー関数 ---
func logDebug(section string, data map[string]any) {
	if !debugMode {
		return
	}
	fmt.Printf("\n[TCGW] %s\n", section)
	for k, v := range data {
		fmt.Printf("  %s: %v\n", k, v)
	}
}
func generateToolCallID() string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		// crypto/randを使用してより安全なランダム生成
		var randomByte [1]byte
		_, _ = cryptorand.Read(randomByte[:])
		b[i] = charset[int(randomByte[0])%len(charset)]
	}
	return "call_" + string(b)
}

func generateResponseID() string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		var randomByte [1]byte
		_, _ = cryptorand.Read(randomByte[:])
		b[i] = charset[int(randomByte[0])%len(charset)]
	}
	return "chatcmpl-" + string(b)
}
func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}
func unescapeXML(s string) string {
	s = strings.ReplaceAll(s, "&apos;", "'")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&amp;", "&")
	return s
}
func inferType(value string) any {
	if value == "true" || value == "false" {
		return value == "true"
	}
	if strings.Contains(value, ".") {
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			return f
		}
	}
	if i, err := strconv.ParseInt(value, 10, 64); err == nil {
		// 32bit環境での安全性を考慮し、int(i) ではなく int64(i) を返す
		return int64(i)
	}
	return value
}

// Contentフィールドから文字列を安全に抽出する
func extractStringContent(content any) string {
	if content == nil {
		return ""
	}
	// stringの場合はそのまま返す
	if str, ok := content.(string); ok {
		return str
	}
	// []ContentPartの場合（構造体として直接渡された場合）
	if parts, ok := content.([]ContentPart); ok {
		var texts []string
		for _, part := range parts {
			if part.Type == "text" {
				texts = append(texts, part.Text)
			}
		}
		return strings.Join(texts, "\n")
	}
	// []interface{}の場合（JSONデシリアライズ後）
	if parts, ok := content.([]any); ok {
		var texts []string
		for _, p := range parts {
			if partMap, ok := p.(map[string]any); ok {
				if partType, _ := partMap["type"].(string); partType == "text" {
					if text, _ := partMap["text"].(string); text != "" {
						texts = append(texts, text)
					}
				}
			}
		}
		if len(texts) > 0 {
			return strings.Join(texts, "\n")
		}
	}
	// その他の型の場合はログ出力して空文字列を返す
	logDebug("Content Type Mismatch", map[string]any{
		"type": fmt.Sprintf("%T", content),
	})
	return ""
}

// Contentフィールドに文字列を安全に設定する
func setStringContent(msg *Message, content string) {
	msg.Content = content
}

// ツール定義 (JSON) を XML 文字列に変換
func generateToolsXML(tools []Tool) string {
	var xml strings.Builder
	for _, tool := range tools {
		xml.WriteString("<tool>\n")
		xml.WriteString(fmt.Sprintf("  <name>%s</name>\n", escapeXML(tool.Function.Name)))
		xml.WriteString(fmt.Sprintf("  <description>%s</description>\n", escapeXML(tool.Function.Description)))
		xml.WriteString("  <parameters>\n")
		// 指示書(セクション12)に基づき、JSON Schemaはエスケープせず、1行のJSONとして埋め込む
		paramsBytes, err := json.Marshal(tool.Function.Parameters)
		if err != nil {
			paramsBytes = []byte("{}") // 堅牢性のため、エラー時は空のオブジェクト
		}
		xml.WriteString("  " + string(paramsBytes) + "\n")
		xml.WriteString("  </parameters>\n</tool>\n")
	}
	return xml.String()
}

// リクエストにツール定義プロンプトを埋め込む
// 既存のツール定義を削除してから最新版を追加（常に最新状態を保証）
func embedToolsIntoPrompt(req *ChatCompletionRequest) {
	if len(req.Tools) == 0 {
		return
	}

	toolsXML := generateToolsXML(req.Tools)
	systemPrompt := strings.ReplaceAll(TOOL_SYSTEM_PROMPT, "{{TOOLS_XML}}", toolsXML)

	if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
		existingContent := extractStringContent(req.Messages[0].Content)

		// ★★★ 改良: 既存のツール定義を削除してから新しいものを追加 ★★★
		cleanedContent := removeToolDefinitions(existingContent)

		// クリーンなコンテンツが空の場合は、ツール定義のみを設定
		if cleanedContent == "" {
			setStringContent(&req.Messages[0], systemPrompt)
		} else {
			setStringContent(&req.Messages[0], systemPrompt+"\n\n"+cleanedContent)
		}
	} else {
		// systemメッセージが存在しない場合は新規作成
		newMessages := make([]Message, 1+len(req.Messages))
		newMessages[0] = Message{Role: "system", Content: systemPrompt}
		copy(newMessages[1:], req.Messages)
		req.Messages = newMessages
	}

	req.Tools = nil      // ツール定義を削除 (Bifrostには送らない)
	req.ToolChoice = nil // Tools が無いのに ToolChoice を送るとプロバイダーでエラーが返されるため、ここで削除しておく
	logDebug("Embedding Tools", map[string]any{
		"System Prompt Len": len(systemPrompt),
		"Messages Count":    len(req.Messages),
	})
}

// システムメッセージから既存のツール定義を削除する
func removeToolDefinitions(content string) string {
	// ツール定義全体（TOOL_SYSTEM_PROMPTの内容）を削除
	// 方法1: <tools>...</tools>とその前後のインストラクションを削除
	lines := strings.Split(content, "\n")
	var cleanedLines []string
	inToolSection := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// ツール定義セクションの開始を検出
		if strings.Contains(trimmed, "You are a helpful AI assistant with access to the following tools:") {
			inToolSection = true
			continue
		}

		// ツール定義セクションの終了を検出
		if inToolSection {
			if strings.Contains(trimmed, "Always use the exact tool names and parameter names as specified.") {
				inToolSection = false
				continue
			}
			continue
		}

		// ツール定義セクション外の行のみを保持
		if trimmed != "" {
			cleanedLines = append(cleanedLines, line)
		}
	}

	result := strings.Join(cleanedLines, "\n")
	return strings.TrimSpace(result)
}

// XML形式のツール呼び出し抽出
func extractXMLToolCalls(text string) []ToolCall {
	fc := reFunctionCalls.FindString(text)
	if fc == "" {
		return nil
	}
	matches := reInvoke.FindAllStringSubmatch(fc, -1)
	if len(matches) == 0 {
		return nil
	}
	var toolCalls []ToolCall
	for _, m := range matches {
		if len(m) < 3 { // m[0]=full, m[1]=name, m[2]=inner
			continue
		}
		toolName := m[1]
		inner := m[2]
		paramMatches := reParameter.FindAllStringSubmatch(inner, -1)
		params := map[string]any{}
		for _, pm := range paramMatches {
			if len(pm) >= 3 { // pm[0]=full, pm[1]=name, pm[2]=value
				// パラメータ値のXMLエスケープを解除 (例: &apos; -> ')
				params[pm[1]] = inferType(unescapeXML(pm[2]))
			}
		}
		paramsJSON, _ := json.Marshal(params)
		toolCalls = append(toolCalls, ToolCall{
			ID:   generateToolCallID(),
			Type: "function",
			Function: ToolCallFunction{
				Name:      toolName,
				Arguments: string(paramsJSON),
			},
		})
	}
	return toolCalls
}

// JSON形式のツール呼び出し抽出 (フォールバック)
func extractJSONToolCalls(text string) []ToolCall {
	j := reJSON.FindString(text)
	if j == "" {
		return nil
	}
	var data map[string]any
	if err := json.Unmarshal([]byte(j), &data); err != nil {
		return nil
	}
	tcs, ok := data["tool_calls"].([]any)
	if !ok {
		return nil
	}
	var toolCalls []ToolCall
	for _, v := range tcs {
		tc, ok := v.(map[string]any)
		if !ok {
			continue
		}
		id, _ := tc["id"].(string)
		if id == "" {
			id = generateToolCallID()
		}
		fn, ok := tc["function"].(map[string]any)
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		args, _ := fn["arguments"].(string)
		toolCalls = append(toolCalls, ToolCall{ID: id, Type: "function", Function: ToolCallFunction{Name: name, Arguments: args}})
	}
	return toolCalls
}

// Markdown JSON形式のツール呼び出し抽出 (フォールバック)
func extractMarkdownToolCalls(text string) []ToolCall {
	ms := reMarkdownJSON.FindAllStringSubmatch(text, -1)
	for _, m := range ms {
		if len(m) >= 2 { // m[0]=full, m[1]=json_content
			var data map[string]any
			if err := json.Unmarshal([]byte(m[1]), &data); err != nil {
				continue
			}
			tcs, ok := data["tool_calls"].([]any)
			if !ok {
				continue
			}
			var toolCalls []ToolCall
			for _, v := range tcs {
				tc, ok := v.(map[string]any)
				if !ok {
					continue
				}
				id, _ := tc["id"].(string)
				if id == "" {
					id = generateToolCallID()
				}
				fn, ok := tc["function"].(map[string]any)
				if !ok {
					continue
				}
				name, _ := fn["name"].(string)
				args, _ := fn["arguments"].(string)
				toolCalls = append(toolCalls, ToolCall{ID: id, Type: "function", Function: ToolCallFunction{Name: name, Arguments: args}})
			}
			if len(toolCalls) > 0 {
				return toolCalls
			}
		}
	}
	return nil
}

// ツール呼び出しの優先抽出 (XML > JSON > Markdown)
func extractToolCalls(text string) []ToolCall {
	if xs := extractXMLToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "XML"})
		return xs
	}
	if xs := extractJSONToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "JSON"})
		return xs
	}
	if xs := extractMarkdownToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Markdown JSON"})
		return xs
	}
	return nil
}

// バックエンドのレスポンスから 'content' 文字列を安全に抽出
func extractContentFromBackendResponse(m map[string]any) string {
	choices, ok := m["choices"].([]any)
	if !ok || len(choices) == 0 {
		logDebug("Content Extraction Failed", map[string]any{
			"reason": "choices field missing or empty",
		})
		return ""
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		logDebug("Content Extraction Failed", map[string]any{
			"reason": "invalid choice structure",
		})
		return ""
	}
	message, ok := choice["message"].(map[string]any)
	if !ok {
		logDebug("Content Extraction Failed", map[string]any{
			"reason": "message field missing",
		})
		return ""
	}
	content, ok := message["content"]
	if !ok || content == nil {
		// tool_callsの場合はcontentがnullなので、これは正常なケース
		return ""
	}
	if str, ok := content.(string); ok {
		return str
	}
	logDebug("Content Extraction Failed", map[string]any{
		"reason":       "content is not a string",
		"content_type": fmt.Sprintf("%T", content),
	})
	return ""
}

// OpenAI互換レスポンスを構築
func buildOpenAIResponse(model, originalContent string, toolCalls []ToolCall) ChatCompletionResponse {
	msg := ResponseMessage{Role: "assistant"}
	var finish string
	if len(toolCalls) > 0 {
		msg.Content = nil // ツール呼び出し時は content は null
		msg.ToolCalls = toolCalls
		finish = "tool_calls"
	} else {
		msg.Content = &originalContent
		msg.ToolCalls = []ToolCall{} // 空配列 (omitemptyにより省略される)
		finish = "stop"
	}
	return ChatCompletionResponse{
		ID:      generateResponseID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: finish,
			},
		},
		Usage: Usage{
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}
}

// Bifrostへリクエスト転送し、JSONレスポンスを返す
func forwardToBifrost(req *ChatCompletionRequest) (map[string]any, error) {
	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("Internal error: failed to marshal request: %v", err)
	}

	logDebug("Forwarding to Bifrost", map[string]any{
		"URL":       bifrostURL + "/v1/chat/completions",
		"Body Size": len(bodyBytes),
		"Timeout":   requestTimeout,
	})

	client := &http.Client{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(requestTimeout)*time.Millisecond)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, bifrostURL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("Internal error: failed to create request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if bifrostApiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+bifrostApiKey)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		// タイムアウト (os.IsTimeout ではなく context.DeadlineExceeded をチェック)
		if errors.Is(err, context.DeadlineExceeded) {
			return map[string]any{"error": map[string]any{"message": fmt.Sprintf("Request timeout after %dms", requestTimeout), "type": "server_error"}}, fmt.Errorf("500")
		}
		// DNS失敗や接続拒否
		if strings.Contains(err.Error(), "connection refused") || strings.Contains(err.Error(), "no such host") {
			return map[string]any{"error": map[string]any{"message": fmt.Sprintf("Backend service unavailable: %v", err), "type": "service_unavailable_error"}}, fmt.Errorf("503")
		}
		// その他ネットワークエラー
		return map[string]any{"error": map[string]any{"message": fmt.Sprintf("Backend service error: %v", err), "type": "server_error"}}, fmt.Errorf("500")
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return map[string]any{"error": map[string]any{"message": fmt.Sprintf("Internal error: failed to read response body: %v", err), "type": "server_error"}}, fmt.Errorf("500")
	}

	logDebug("Bifrost Response Received", map[string]any{
		"Status Code": resp.StatusCode,
		"Body Size":   len(body),
	})

	if resp.StatusCode >= 400 {
		var backendErr map[string]any
		if json.Unmarshal(body, &backendErr) == nil {
			// Bifrostからのエラーをそのまま転送
			return backendErr, fmt.Errorf("%d", resp.StatusCode)
		} else {
			// BifrostがJSONでないエラーを返した場合
			return map[string]any{"error": map[string]any{"message": "Invalid response from backend", "type": "server_error"}}, fmt.Errorf("502")
		}
	}

	var backendResp map[string]any
	if err := json.Unmarshal(body, &backendResp); err != nil {
		return map[string]any{"error": map[string]any{"message": "Invalid response from backend (JSON parse failed)", "type": "server_error"}}, fmt.Errorf("502")
	}
	return backendResp, nil
}

// --- Ginハンドラー群 ---

// エミュレートモード: ツール呼び出しをXML形式でエミュレート
func handleChatCompletionsEmulate(c *gin.Context) {
	var req ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, ErrorResponse{Error: ErrorDetail{
			Message: fmt.Sprintf("Invalid JSON: %v", err),
			Type:    "invalid_request_error",
			Code:    stringPtr("invalid_request"),
		}})
		return
	}

	if req.Stream {
		c.JSON(501, ErrorResponse{Error: ErrorDetail{
			Message: "Streaming is not currently supported",
			Type:    "invalid_request_error",
		}})
		return
	}

	logDebug("Request Received (Emulate Mode)", map[string]any{
		"Model":         req.Model,
		"Tool Count":    len(req.Tools),
		"Message Count": len(req.Messages),
	})

	embedToolsIntoPrompt(&req)
	backendResp, ferr := forwardToBifrost(&req)
	if ferr != nil {
		code := 500
		if s, err := strconv.Atoi(ferr.Error()); err == nil {
			code = s
		}
		c.JSON(code, backendResp)
		return
	}

	content := extractContentFromBackendResponse(backendResp)
	toolCalls := extractToolCalls(content)

	// 部分的な上書きを実行
	patchedResp := patchOpenAIResponse(backendResp, toolCalls)
	if patchedResp == nil {
		// フォールバック: 従来の完全書き換え
		resp := buildOpenAIResponse(req.Model, content, toolCalls)
		c.JSON(200, resp)
		return
	}

	logDebug("Response Patched (Emulate Mode)", map[string]any{
		"Tool Calls Count": len(toolCalls),
		"Finish Reason":    patchedResp["choices"].([]any)[0].(map[string]any)["finish_reason"],
	})

	c.JSON(200, patchedResp)
}

// stringのポインタを返すヘルパー関数
func stringPtr(s string) *string {
	return &s
}

// func handleChatCompletionsEmulate(c *gin.Context) {
// 	var req ChatCompletionRequest
// 	if err := c.ShouldBindJSON(&req); err != nil {
// 		c.JSON(400, ErrorResponse{Error: struct {
// 			Message string `json:"message"`
// 			Type    string `json:"type"`
// 			Code    string `json:"code,omitempty"`
// 		}{Message: fmt.Sprintf("Invalid JSON: %v", err), Type: "invalid_request_error", Code: "invalid_request"}})
// 		return
// 	}

// 	// ストリーミングは非対応 (指示書セクション15に基づき 501 Not Implemented)
// 	if req.Stream {
// 		c.JSON(501, ErrorResponse{Error: struct {
// 			Message string `json:"message"`
// 			Type    string `json:"type"`
// 			Code    string `json:"code,omitempty"`
// 		}{Message: "Streaming is not currently supported", Type: "invalid_request_error"}})
// 		return
// 	}

// 	logDebug("Request Received (Emulate Mode)", map[string]any{
// 		"Model":         req.Model,
// 		"Tool Count":    len(req.Tools),
// 		"Message Count": len(req.Messages),
// 		"Has Stream":    req.Stream,
// 	})

// 	// ステップ1: ツール定義をプロンプトに埋め込む
// 	embedToolsIntoPrompt(&req)

// 	// ステップ2: Bifrostに転送
// 	backendResp, ferr := forwardToBifrost(&req)
// 	if ferr != nil {
// 		code := 500
// 		if s, err := strconv.Atoi(ferr.Error()); err == nil {
// 			code = s // 503, 502, 4xx など
// 		}
// 		c.JSON(code, backendResp)
// 		return
// 	}

// 	// ステップ3: Bifrostレスポンスからコンテンツを抽出
// 	content := extractContentFromBackendResponse(backendResp)

// 	// ステップ4: コンテンツからツール呼び出しを抽出
// 	toolCalls := extractToolCalls(content)
// 	logDebug("Tool Calls Extracted", map[string]any{"Count": len(toolCalls)})

// 	// ステップ5: OpenAI互換レスポンスを構築
// 	resp := buildOpenAIResponse(req.Model, content, toolCalls)
// 	logDebug("Response Generated (Emulate Mode)", map[string]any{
// 		"Finish Reason":    resp.Choices[0].FinishReason,
// 		"Tool Calls Count": len(toolCalls),
// 		"Response ID":      resp.ID,
// 	})

// 	c.JSON(200, resp)
// }

// バックエンドレスポンスを部分的に上書きしてOpenAI互換にする
func patchOpenAIResponse(backendResp map[string]any, toolCalls []ToolCall) map[string]any {
	choices, ok := backendResp["choices"].([]any)
	if !ok || len(choices) == 0 {
		logDebug("Patch Failed", map[string]any{
			"reason": "choices field invalid",
		})
		return nil
	}

	choice, ok := choices[0].(map[string]any)
	if !ok {
		logDebug("Patch Failed", map[string]any{
			"reason": "choice[0] is not a map",
		})
		return nil
	}

	message, ok := choice["message"].(map[string]any)
	if !ok {
		// messageが存在しない場合は新規作成
		message = map[string]any{"role": "assistant"}
		choice["message"] = message
		logDebug("Patch: Created new message", map[string]any{})
	}

	// ツール呼び出しの有無で分岐
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		message["content"] = nil
		choice["finish_reason"] = "tool_calls"
		logDebug("Patch: Added tool_calls", map[string]any{
			"count": len(toolCalls),
		})
	} else {
		delete(message, "tool_calls")
		choice["finish_reason"] = "stop"
	}

	backendResp["choices"] = []any{choice}
	return backendResp
}

// パススルーモード: リクエストをそのままBifrostに転送（ネイティブTool Calling使用）
func handleChatCompletionsPassthrough(c *gin.Context) {
	var req ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, ErrorResponse{Error: ErrorDetail{
			Message: fmt.Sprintf("Invalid JSON: %v", err),
			Type:    "invalid_request_error",
			Code:    stringPtr("invalid_request"),
		}})
		return
	}

	logDebug("Request Received (Passthrough Mode)", map[string]any{
		"Model":         req.Model,
		"Tool Count":    len(req.Tools),
		"Message Count": len(req.Messages),
		"Has Stream":    req.Stream,
	})

	// ツール定義の埋め込みは行わず、リクエストをそのまま転送
	backendResp, ferr := forwardToBifrost(&req)
	if ferr != nil {
		code := 500
		if s, err := strconv.Atoi(ferr.Error()); err == nil {
			code = s
		}
		c.JSON(code, backendResp)
		return
	}

	// ★★★ 追加: レスポンスをクリーンアップ ★★★
	cleanedResp := cleanupToolCallsInResponse(backendResp)

	logDebug("Response Forwarded (Passthrough Mode)", map[string]any{
		"Status": "Success",
	})

	// クリーンアップしたレスポンスを返却
	c.JSON(200, cleanedResp)
}

// レスポンスから空のtool_calls配列を削除し、finish_reasonも適切に調整する
func cleanupToolCallsInResponse(resp map[string]any) map[string]any {
	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		return resp
	}

	for i := range choices {
		choice, ok := choices[i].(map[string]any)
		if !ok {
			continue
		}

		message, ok := choice["message"].(map[string]any)
		if !ok {
			continue
		}

		// tool_callsが存在する場合
		if toolCalls, exists := message["tool_calls"]; exists {
			// 空配列または nil の場合は削除
			shouldDelete := false
			if toolCalls == nil {
				shouldDelete = true
			} else if tcArray, ok := toolCalls.([]any); ok && len(tcArray) == 0 {
				shouldDelete = true
			}

			if shouldDelete {
				delete(message, "tool_calls")
				// finish_reasonがtool_callsの場合、stopに修正
				if finishReason, ok := choice["finish_reason"].(string); ok && finishReason == "tool_calls" {
					choice["finish_reason"] = "stop"
					logDebug("Cleanup: Changed finish_reason", map[string]any{
						"from": "tool_calls",
						"to":   "stop",
					})
				}
			}
		}
	}

	return resp
}

func handleHealthCheck(c *gin.Context) {
	health := gin.H{
		"status":    "ok",
		"service":   "tcgw",
		"version":   config.VERSION,
		"mode":      "dual-port",
		"timestamp": time.Now().Unix(),
	}

	// Bifrost接続チェック（オプショナル）
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(bifrostURL + "/health")
	if err != nil || resp.StatusCode != 200 {
		health["bifrost_status"] = "unreachable"
		health["status"] = "degraded"
	} else {
		health["bifrost_status"] = "ok"
	}
	if resp != nil {
		resp.Body.Close()
	}

	statusCode := 200
	if health["status"] == "degraded" {
		statusCode = 503
	}

	c.JSON(statusCode, health)
}

// --- サーバ起動 ---
func main() {
	initConfig()

	if !debugMode {
		gin.SetMode(gin.ReleaseMode)
	}

	// エミュレートモード用サーバー
	emulateRouter := gin.Default()
	emulateRouter.Use(cors.Default())
	v1Emulate := emulateRouter.Group("/v1")
	v1Emulate.POST("/chat/completions", handleChatCompletionsEmulate)
	emulateRouter.GET("/health", handleHealthCheck)

	// パススルーモード用サーバー
	passthroughRouter := gin.Default()
	passthroughRouter.Use(cors.Default())
	v1Passthrough := passthroughRouter.Group("/v1")
	v1Passthrough.POST("/chat/completions", handleChatCompletionsPassthrough)
	passthroughRouter.GET("/health", handleHealthCheck)

	// 2つのサーバーを同時起動
	go func() {
		if err := emulateRouter.Run(emulatePort); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to start emulate server: %v\n", err)
			os.Exit(1)
		}
	}()

	if err := passthroughRouter.Run(passthroughPort); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start passthrough server: %v\n", err)
		os.Exit(1)
	}
}
