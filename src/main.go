/**
 * main.go
 *
 * TCGW - 任意のLLMにOpenAI Function Calling互換の「ツール呼び出し」能力をエミュレートするGo製プロキシ
 * このmain.goは、「.envの設定・Bifrost連携・OpenAI形式APIでの受け付け・XMLツール抽出・OpenAI形式レスポンス化」まで完全に担います。
 *
 * 動作モード
 * - Tool Calling Emulation: ツール呼び出しをXML形式でエミュレート
 * - ネイティブTool Calling対応モデルを使用する場合は、Bifrostに直接リクエストすること
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

MANDATORY: Every response must be either a tool call OR a final conversational answer. Empty or null responses are STRICTLY FORBIDDEN.

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
6. When no more tools are needed, you MUST provide a final conversational response in Japanese. Empty responses are FORBIDDEN.
7. Your final conversational response MUST be in Japanese (日本語).
8. CRITICAL: You MUST provide a final response when no more tools are needed. Empty responses are FORBIDDEN.
   - If all tools have been executed, you MUST output a conversational answer
   - NEVER leave the response empty or output only whitespace
9. Follow the EXAMPLE WORKFLOW below exactly - this demonstrates the required behavior of calling tools immediately without any explanation.

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

// ========================================
// 正規表現の事前コンパイル（パフォーマンス最適化）
// すべての正規表現を起動時に一度だけコンパイル
// ========================================

var (
	// GPT-OSS
	regexGPTOSS = regexp.MustCompile(`<\|channel\|>(commentary|analysis)\s+to=(?:functions\.)?([a-zA-Z0-9_]+)(?:\s+<\|constrain\|>[a-zA-Z0-9_-]+)?(?:\s+<\|message\|>)?(.*?)(?:<\|call\|>|$)`)

	// Hermes 2 Pro - 複雑な開始パターン
	regexHermes2ProOpen = regexp.MustCompile(`(?:(<|\[)?)?` +
		`(<tool_call>|<functioncall>|<function>|<tool>|<tools>|<response>|<json>|<xml>|<JSON>)?` +
		`\s*` +
		`(?:<name>([^<]+)</name>)?` +
		`(?:<function>([^<(]+))?` +
		`(?:<function>([^<]+))?`)

	// Functionary v3.2
	regexFunctionaryV32 = regexp.MustCompile(`>>>(\w+)`)

	// Functionary v3.1 Llama 3.1
	regexFunctionaryV31Llama31 = regexp.MustCompile(`<function=([^>]+)>`)

	// Llama 3.x
	regexLlama3X = regexp.MustCompile(`\{"type":\s*"function",\s*"name":\s*"([^"]+)",\s*"parameters":\s*`)

	// DeepSeek V3.1
	regexDeepSeekV31Function = regexp.MustCompile(`<｜tool▁call▁begin｜>([^<｜]*)<｜tool▁sep｜>`)

	// DeepSeek R1
	regexDeepSeekR1Function = regexp.MustCompile(`<｜tool▁call▁begin｜>([^<｜]*)<｜function▁tool▁sep｜>|<｜tool▁call▁begin｜><｜function▁tool▁sep｜>`)
)

// --- グローバル変数 (設定) ---
var bifrostURL string
var emulatePort string // エミュレートモード用ポート
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

	fmt.Println("🌉 TCGW Proxy Server")
	if debugMode {
		fmt.Printf("[TCGW] Server Configuration\n Port: %s\n Bifrost URL: %s\n Debug Mode: true\n Request Timeout: %dms\n",
			strings.TrimPrefix(emulatePort, ":"),
			bifrostURL,
			requestTimeout)
	}

	fmt.Println("[TCGW] Server Starting")
	fmt.Printf(" Tool Calling Emulation: 0.0.0.0%s\n", emulatePort)
	fmt.Printf(" BIFROST: %s\n", bifrostURL)
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

// extractToolCalls はLLMの出力からツール呼び出しを抽出
// llama.cpp式の多段階パース戦略：モデルファミリー別 → 標準形式 → ジェネリック
func extractToolCalls(text string) []ToolCall {
	// Phase 1: モデルファミリー別パーサー（特定モデルの独自形式）
	// llama.cppのcommon_chat_templates_apply_jinjaの検出順序に基づく

	// DeepSeek V3.1
	if xs := extractDeepSeekV31ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "DeepSeek V3.1"})
		return xs
	}

	// DeepSeek R1
	if xs := extractDeepSeekR1ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "DeepSeek R1"})
		return xs
	}

	// Command R7B
	if xs := extractCommandR7BToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Command R7B"})
		return xs
	}

	// Granite (IBM)
	if xs := extractGraniteToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Granite"})
		return xs
	}

	// GLM 4.5（Hermes 2 Proより先にチェック - 両方とも<tool_call>を使用）
	if xs := extractGLM45ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "GLM 4.5"})
		return xs
	}

	// Qwen3-Coder XML（Hermes 2 Proより先にチェック）
	if xs := extractQwen3CoderXMLToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Qwen3-Coder XML"})
		return xs
	}

	// Xiaomi MiMo（Hermes 2 Proより先にチェック）
	if xs := extractXiaomiMiMoToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Xiaomi MiMo"})
		return xs
	}

	// Hermes 2 Pro, Qwen 2.5 Instruct
	if xs := extractHermes2ProToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Hermes 2 Pro"})
		return xs
	}

	// GPT-OSS
	if xs := extractGPTOSSToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "GPT-OSS"})
		return xs
	}

	// Seed-OSS
	if xs := extractSeedOSSToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Seed-OSS"})
		return xs
	}

	// Nemotron v2
	if xs := extractNemotronV2ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Nemotron v2"})
		return xs
	}

	// Apertus
	if xs := extractApertusToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Apertus"})
		return xs
	}

	// LFM2
	if xs := extractLFM2ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "LFM2"})
		return xs
	}

	// MiniMax-M2
	if xs := extractMiniMaxM2ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "MiniMax-M2"})
		return xs
	}

	// Kimi K2
	if xs := extractKimiK2ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Kimi K2"})
		return xs
	}

	// Apriel 1.5
	if xs := extractApriel15ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Apriel 1.5"})
		return xs
	}

	// Functionary v3.2
	if xs := extractFunctionaryV32ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Functionary v3.2"})
		return xs
	}

	// Firefunction v2
	if xs := extractFirefunctionV2ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Firefunction v2"})
		return xs
	}

	// Functionary v3.1 Llama 3.1
	if xs := extractFunctionaryV31Llama31ToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Functionary v3.1 Llama 3.1"})
		return xs
	}

	// Llama 3.x
	if xs := extractLlama3XToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Llama 3.x"})
		return xs
	}

	// Magistral
	if xs := extractMagistralToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Magistral"})
		return xs
	}

	// Mistral Nemo
	if xs := extractMistralNemoToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Mistral Nemo"})
		return xs
	}

	// Phase 2: 標準形式パーサー（既存のTCGW形式）

	// XML形式の検出
	if xs := extractXMLToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "XML"})
		return xs
	}

	// JSON形式の検出
	if xs := extractJSONToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "JSON"})
		return xs
	}

	// Markdown JSON形式の検出
	if xs := extractMarkdownToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Markdown JSON"})
		return xs
	}

	// Phase 3: ジェネリックパーサー（最後の砦）

	// 汎用JSON形式の検出
	if xs := extractGenericToolCalls(text); len(xs) > 0 {
		logDebug("Tool Call Extraction", map[string]any{"Format": "Generic JSON"})
		return xs
	}

	// どのパーサーでも検出できなかった場合
	return nil
}

// extractGPTOSSToolCalls は GPT-OSS 独自形式のツール呼び出しを抽出
// 形式: <|start|>assistant<|channel|>commentary to=functionName <|constrain|>json<|message|>{JSON}<|call|>
func extractGPTOSSToolCalls(text string) []ToolCall {
	// GPT-OSS形式の正規表現パターン
	// <|channel|>commentary to=functionName または <|channel|>analysis to=functionName
	// ドット区切りの関数名に対応（例: functions.calculatePrice）

	matches := regexGPTOSS.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}

	var toolCalls []ToolCall
	for _, match := range matches {
		if len(match) < 4 {
			continue
		}

		channelType := match[1] // commentary or analysis
		functionName := match[2]
		argsText := strings.TrimSpace(match[3])

		// <|message|> タグが含まれている場合は除去
		argsText = strings.TrimPrefix(argsText, "<|message|>")
		argsText = strings.TrimSpace(argsText)

		// 空のツール名はスキップ
		if functionName == "" {
			continue
		}

		// JSON引数のパース
		var argsMap map[string]any
		if argsText != "" {
			// JSONブロックを抽出（中括弧で囲まれた部分）
			jsonStart := strings.Index(argsText, "{")
			jsonEnd := strings.LastIndex(argsText, "}")

			if jsonStart != -1 && jsonEnd != -1 && jsonEnd > jsonStart {
				jsonStr := argsText[jsonStart : jsonEnd+1]
				if err := json.Unmarshal([]byte(jsonStr), &argsMap); err == nil {
					argsBytes, _ := json.Marshal(argsMap)

					toolCalls = append(toolCalls, ToolCall{
						ID:       generateToolCallID(),
						Type:     "function",
						Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
					})

					logDebug("GPT-OSS Tool Call Detected", map[string]any{
						"Channel":  channelType,
						"Function": functionName,
						"Args":     string(argsBytes),
					})
				}
			}
		}
	}

	return toolCalls
}

// extractHermes2ProToolCalls は Hermes 2 Pro 形式のツール呼び出しを抽出
// llama.cppの実装を忠実に移植
func extractHermes2ProToolCalls(text string) []ToolCall {
	// Hermes 2 Pro形式の複雑な正規表現パターン
	// llama.cppの open_regex に対応
	matches := regexHermes2ProOpen.FindAllStringSubmatchIndex(text, -1)

	if len(matches) == 0 {
		return nil
	}

	var toolCalls []ToolCall

	for _, match := range matches {
		// match[0], match[1]: 全体マッチ
		// match[2], match[3]: group 1 (block_start)
		// match[4], match[5]: group 2 (open_tag)
		// match[6], match[7]: group 3 (name in <name>...</name>)
		// match[8], match[9]: group 4 (function name)
		// match[10], match[11]: group 5 (function name alternative)

		if len(match) < 12 {
			continue
		}

		var functionName string
		var openTag string
		var closeTag string
		var jsonStart int

		// open_tag の取得
		if match[4] != -1 && match[5] != -1 {
			openTag = text[match[4]:match[5]]
			// close_tag を構築（例: <tool_call> → </tool_call>）
			if len(openTag) > 1 {
				closeTag = "</" + openTag[1:] + ">"
			}
		}

		// パターン1: <name>functionName</name> 形式
		if match[6] != -1 && match[7] != -1 {
			functionName = strings.TrimSpace(text[match[6]:match[7]])
			jsonStart = match[7]
		} else if match[8] != -1 && match[9] != -1 {
			// パターン2: <function>functionName 形式
			functionName = strings.TrimSpace(text[match[8]:match[9]])
			jsonStart = match[9]
		} else if match[10] != -1 && match[11] != -1 {
			// パターン3: 代替の function name 形式
			functionName = strings.TrimSpace(text[match[10]:match[11]])
			jsonStart = match[11]
		} else {
			// 関数名が見つからない場合、JSONオブジェクトから抽出を試みる
			jsonStart = match[1] // 全体マッチの終了位置から開始
		}

		// 関数名が空の場合はスキップ
		if functionName == "" && openTag == "" {
			continue
		}

		// JSON引数の抽出
		remainingText := text[jsonStart:]

		// closeTagがある場合はそこまでを抽出
		var jsonText string
		if closeTag != "" {
			closeIdx := strings.Index(remainingText, closeTag)
			if closeIdx != -1 {
				jsonText = remainingText[:closeIdx]
			} else {
				jsonText = remainingText
			}
		} else {
			jsonText = remainingText
		}

		// JSONブロックを抽出
		jsonText = strings.TrimSpace(jsonText)
		jsonStartIdx := strings.Index(jsonText, "{")
		jsonEndIdx := strings.LastIndex(jsonText, "}")

		if jsonStartIdx == -1 || jsonEndIdx == -1 || jsonEndIdx <= jsonStartIdx {
			continue
		}

		jsonStr := jsonText[jsonStartIdx : jsonEndIdx+1]

		// JSONパース
		var toolCallData map[string]any
		if err := json.Unmarshal([]byte(jsonStr), &toolCallData); err != nil {
			continue
		}

		// 関数名がまだ取得できていない場合、JSONから取得
		if functionName == "" {
			if name, ok := toolCallData["name"].(string); ok {
				functionName = name
			} else if fn, ok := toolCallData["function"].(string); ok {
				functionName = fn
			}
		}

		// 関数名が依然として空の場合はスキップ
		if functionName == "" {
			continue
		}

		// 引数の取得
		var argsBytes []byte
		if args, exists := toolCallData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				argsBytes = []byte(argsStr)
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			// JSONオブジェクト全体を引数として使用（nameやfunctionキーを除外）
			filteredArgs := make(map[string]any)
			for k, v := range toolCallData {
				if k != "name" && k != "function" {
					filteredArgs[k] = v
				}
			}
			if len(filteredArgs) > 0 {
				argsBytes, _ = json.Marshal(filteredArgs)
			} else {
				argsBytes = []byte("{}")
			}
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Hermes 2 Pro Tool Call Detected", map[string]any{
			"OpenTag":  openTag,
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractFunctionaryV32ToolCalls は Functionary v3.2 形式のツール呼び出しを抽出
// llama.cppの実装を忠実に移植
// 形式: >>>functionName\n{"arg1": "value1"}<<< または >>>python\ncode<<<
func extractFunctionaryV32ToolCalls(text string) []ToolCall {
	// Functionary v3.2形式の正規表現パターン
	// >>> で開始（3つの>）、<<< で終了（3つの<）
	closePattern := `<<<`
	matches := regexFunctionaryV32.FindAllStringSubmatchIndex(text, -1)

	if len(matches) == 0 {
		return nil
	}

	var toolCalls []ToolCall

	for _, match := range matches {
		// match[0], match[1]: 全体マッチ（>>>functionName）
		// match[2], match[3]: group 1 (functionName)

		if len(match) < 4 {
			continue
		}

		atStart := match[0] == 0
		functionName := strings.TrimSpace(text[match[2]:match[3]])

		// 関数名の末尾に '(' がある場合は削除
		if len(functionName) > 0 && functionName[len(functionName)-1] == '(' {
			functionName = strings.TrimRight(functionName, "(")
		}

		// 開始位置で "all" または "python" の場合はスキップ
		if atStart && (functionName == "all" || functionName == "python") {
			continue
		}

		// 空の関数名はスキップ
		if functionName == "" {
			continue
		}

		// 引数部分の抽出（>>> の後から <<< まで）
		argsStart := match[1] // >>> の終了位置
		remainingText := text[argsStart:]

		// <<< を探す
		closeIdx := strings.Index(remainingText, closePattern)
		if closeIdx == -1 {
			// 閉じタグが見つからない場合は残り全体
			closeIdx = len(remainingText)
		}

		argsText := strings.TrimSpace(remainingText[:closeIdx])

		// Pythonコードの特殊処理
		if functionName == "python" && !strings.HasPrefix(argsText, "{") {
			// Raw Pythonコード: JSON形式でラップ
			codeJSON := map[string]any{
				"code": argsText,
			}
			argsBytes, _ := json.Marshal(codeJSON)

			toolCalls = append(toolCalls, ToolCall{
				ID:       generateToolCallID(),
				Type:     "function",
				Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
			})

			logDebug("Functionary v3.2 Tool Call Detected (Python)", map[string]any{
				"Function": functionName,
				"Code":     argsText,
			})
			continue
		}

		// JSON引数のパース
		jsonStart := strings.Index(argsText, "{")
		jsonEnd := strings.LastIndex(argsText, "}")

		if jsonStart == -1 || jsonEnd == -1 || jsonEnd <= jsonStart {
			// JSONが見つからない場合は空オブジェクト
			toolCalls = append(toolCalls, ToolCall{
				ID:       generateToolCallID(),
				Type:     "function",
				Function: ToolCallFunction{Name: functionName, Arguments: "{}"},
			})
			continue
		}

		jsonStr := argsText[jsonStart : jsonEnd+1]

		// JSON検証
		var argsMap map[string]any
		if err := json.Unmarshal([]byte(jsonStr), &argsMap); err != nil {
			continue
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Functionary v3.2 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractDeepSeekV31ToolCalls は DeepSeek V3.1 形式のツール呼び出しを抽出
// 形式: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>functionName<｜tool▁sep｜>{JSON}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
func extractDeepSeekV31ToolCalls(text string) []ToolCall {
	// DeepSeek V3.1の特殊トークン（全角文字を含む）
	const (
		toolCallsBegin = "<｜tool▁calls▁begin｜>"
		toolCallBegin  = "<｜tool▁call▁begin｜>"
		toolSep        = "<｜tool▁sep｜>"
		toolCallEnd    = "<｜tool▁call▁end｜>"
		toolCallsEnd   = "<｜tool▁calls▁end｜>"
	)

	// 複数のバリエーションに対応（llama.cppのtoolcalls_beginパターン）
	toolCallsBeginVariants := []string{
		"<｜tool▁calls▁begin｜>",
		"<tool calls begin>",
		"<toolcalls>",
	}

	// いずれかのバリエーションが存在するか確認
	hasBegin := false
	for _, variant := range toolCallsBeginVariants {
		if strings.Contains(text, variant) {
			hasBegin = true
			break
		}
	}

	if !hasBegin {
		return nil
	}

	// toolCallsBeginからtoolCallsEndまでの範囲を抽出
	startIdx := -1
	for _, variant := range toolCallsBeginVariants {
		idx := strings.Index(text, variant)
		if idx != -1 {
			startIdx = idx
			break
		}
	}

	if startIdx == -1 {
		return nil
	}

	endIdx := strings.Index(text[startIdx:], toolCallsEnd)
	var toolCallsText string
	if endIdx == -1 {
		// 終了タグが見つからない場合は残り全体
		toolCallsText = text[startIdx:]
	} else {
		toolCallsText = text[startIdx : startIdx+endIdx+len(toolCallsEnd)]
	}

	var toolCalls []ToolCall

	// 正規表現でツール呼び出しを抽出
	// パターン: <｜tool▁call▁begin｜>functionName<｜tool▁sep｜>
	matches := regexDeepSeekV31Function.FindAllStringSubmatchIndex(toolCallsText, -1)

	for _, match := range matches {
		// match[2], match[3]: 関数名のキャプチャグループ
		if len(match) < 4 {
			continue
		}

		functionName := strings.TrimSpace(toolCallsText[match[2]:match[3]])

		// 空の関数名はスキップ
		if functionName == "" {
			continue
		}

		// JSON引数の抽出（<｜tool▁sep｜>から<｜tool▁call▁end｜>まで）
		jsonStart := match[1] // <｜tool▁sep｜>の直後
		remainingText := toolCallsText[jsonStart:]

		jsonEnd := strings.Index(remainingText, toolCallEnd)
		if jsonEnd == -1 {
			continue
		}

		jsonText := strings.TrimSpace(remainingText[:jsonEnd])

		// JSON引数のパース
		var argsMap map[string]any
		if jsonText != "" {
			if err := json.Unmarshal([]byte(jsonText), &argsMap); err != nil {
				continue
			}
		} else {
			argsMap = make(map[string]any)
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("DeepSeek V3.1 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractDeepSeekR1ToolCalls は DeepSeek R1 形式のツール呼び出しを抽出
// 形式: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>functionName<｜function▁tool▁sep｜>{JSON}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
func extractDeepSeekR1ToolCalls(text string) []ToolCall {
	// DeepSeek R1の特殊トークン
	const (
		toolCallsBegin  = "<｜tool▁calls▁begin｜>"
		toolCallBegin   = "<｜tool▁call▁begin｜>"
		functionToolSep = "<｜function▁tool▁sep｜>"
		toolCallEnd     = "<｜tool▁call▁end｜>"
		toolCallsEnd    = "<｜tool▁calls▁end｜>"
	)

	// 複数のバリエーションに対応
	toolCallsBeginVariants := []string{
		"<｜tool▁calls▁begin｜>",
		"<tool calls begin>",
		"<｜tool▁calls▁begin｜>",
		"<toolcalls>",
	}

	// いずれかのバリエーションが存在するか確認
	hasBegin := false
	for _, variant := range toolCallsBeginVariants {
		if strings.Contains(text, variant) {
			hasBegin = true
			break
		}
	}

	if !hasBegin {
		return nil
	}

	// toolCallsBeginからtoolCallsEndまでの範囲を抽出
	startIdx := -1
	for _, variant := range toolCallsBeginVariants {
		idx := strings.Index(text, variant)
		if idx != -1 {
			startIdx = idx
			break
		}
	}

	if startIdx == -1 {
		return nil
	}

	endIdx := strings.Index(text[startIdx:], toolCallsEnd)
	var toolCallsText string
	if endIdx == -1 {
		toolCallsText = text[startIdx:]
	} else {
		toolCallsText = text[startIdx : startIdx+endIdx+len(toolCallsEnd)]
	}

	var toolCalls []ToolCall

	// 正規表現でツール呼び出しを抽出
	// パターン1: <｜tool▁call▁begin｜>functionName<｜function▁tool▁sep｜>
	// パターン2: <｜tool▁call▁begin｜><｜function▁tool▁sep｜> (関数名なし)
	matches := regexDeepSeekR1Function.FindAllStringSubmatchIndex(toolCallsText, -1)

	for _, match := range matches {
		// match[0], match[1]: 全体マッチ
		// match[2], match[3]: 関数名のキャプチャグループ（存在する場合）

		var functionName string
		if len(match) >= 4 && match[2] != -1 && match[3] != -1 {
			functionName = strings.TrimSpace(toolCallsText[match[2]:match[3]])
		}

		// 関数名が空の場合はスキップ（パターン2の場合も）
		if functionName == "" {
			continue
		}

		// JSON引数の抽出（<｜function▁tool▁sep｜>から<｜tool▁call▁end｜>まで）
		jsonStart := match[1] // マッチ全体の終了位置
		remainingText := toolCallsText[jsonStart:]

		jsonEnd := strings.Index(remainingText, toolCallEnd)
		if jsonEnd == -1 {
			continue
		}

		jsonText := strings.TrimSpace(remainingText[:jsonEnd])

		// JSON引数のパース
		var argsMap map[string]any
		if jsonText != "" {
			if err := json.Unmarshal([]byte(jsonText), &argsMap); err != nil {
				continue
			}
		} else {
			argsMap = make(map[string]any)
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("DeepSeek R1 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractCommandR7BToolCalls は Command R7B 形式のツール呼び出しを抽出
// 形式: <|START_ACTION|>[{"tool_name": "func", "tool_call_id": "id", "parameters": {...}}]<|END_ACTION|>
func extractCommandR7BToolCalls(text string) []ToolCall {
	// Command R7Bの特殊トークン
	const (
		startAction = "<|START_ACTION|>"
		endAction   = "<|END_ACTION|>"
	)

	// START_ACTIONの検出
	startIdx := strings.Index(text, startAction)
	if startIdx == -1 {
		return nil
	}

	// END_ACTIONの検出
	endIdx := strings.Index(text[startIdx:], endAction)
	var actionText string
	if endIdx == -1 {
		actionText = text[startIdx+len(startAction):]
	} else {
		actionText = text[startIdx+len(startAction) : startIdx+endIdx]
	}

	actionText = strings.TrimSpace(actionText)

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(actionText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// tool_nameの取得
		functionName, ok := tcData["tool_name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// tool_call_idの取得（オプション）
		toolCallID := ""
		if id, ok := tcData["tool_call_id"].(string); ok {
			toolCallID = id
		}

		// parametersの取得
		var argsBytes []byte
		if params, exists := tcData["parameters"]; exists {
			if paramsMap, ok := params.(map[string]any); ok {
				argsBytes, _ = json.Marshal(paramsMap)
			} else if paramsStr, ok := params.(string); ok {
				// 文字列の場合はそのまま使用
				argsBytes = []byte(paramsStr)
			} else {
				argsBytes, _ = json.Marshal(params)
			}
		} else {
			argsBytes = []byte("{}")
		}

		// IDが指定されている場合はそれを使用、なければ生成
		if toolCallID == "" {
			toolCallID = generateToolCallID()
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       toolCallID,
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Command R7B Tool Call Detected", map[string]any{
			"Function": functionName,
			"ID":       toolCallID,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractGraniteToolCalls は Granite (IBM) 形式のツール呼び出しを抽出
// 形式: <tool_call>[{"name": "func", "arguments": {...}}]
func extractGraniteToolCalls(text string) []ToolCall {
	// Graniteの特殊トークン
	const toolCallTag = "<tool_call>"

	// tool_callタグの検出
	tagIdx := strings.Index(text, toolCallTag)
	if tagIdx == -1 {
		return nil
	}

	// JSON配列の抽出（<tool_call>の後から）
	jsonText := strings.TrimSpace(text[tagIdx+len(toolCallTag):])

	// JSON配列の開始を探す
	if !strings.HasPrefix(jsonText, "[") {
		return nil
	}

	// JSON配列の終了を探す
	jsonEnd := strings.LastIndex(jsonText, "]")
	if jsonEnd == -1 {
		return nil
	}

	jsonText = jsonText[:jsonEnd+1]

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(jsonText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// nameの取得
		functionName, ok := tcData["name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// argumentsの取得
		var argsBytes []byte
		if args, exists := tcData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Granite Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractGLM45ToolCalls は GLM 4.5 形式のツール呼び出しを抽出
// 形式: <tool_call><arg_key>param1</arg_key><arg_value>value1</arg_value>...</tool_call>
func extractGLM45ToolCalls(text string) []ToolCall {
	// GLM 4.5のXML形式タグ
	const (
		toolCallStart = "<tool_call>"
		toolCallEnd   = "</tool_call>"
		argKeyStart   = "<arg_key>"
		argKeyEnd     = "</arg_key>"
		argValueStart = "<arg_value>"
		argValueEnd   = "</arg_value>"
	)

	// tool_callタグの検出
	if !strings.Contains(text, toolCallStart) {
		return nil
	}

	var toolCalls []ToolCall

	// 複数のtool_callを抽出
	searchPos := 0
	for {
		// tool_callの開始を探す
		startIdx := strings.Index(text[searchPos:], toolCallStart)
		if startIdx == -1 {
			break
		}
		startIdx += searchPos

		// tool_callの終了を探す
		endIdx := strings.Index(text[startIdx:], toolCallEnd)
		if endIdx == -1 {
			break
		}
		endIdx += startIdx

		// 1つのツール呼び出しブロック
		toolCallText := text[startIdx+len(toolCallStart) : endIdx]

		// arg_key/arg_valueペアを抽出
		args := make(map[string]any)
		var functionName string

		argSearchPos := 0
		for {
			// arg_keyの開始を探す
			keyStartIdx := strings.Index(toolCallText[argSearchPos:], argKeyStart)
			if keyStartIdx == -1 {
				break
			}
			keyStartIdx += argSearchPos

			// arg_keyの終了を探す
			keyEndIdx := strings.Index(toolCallText[keyStartIdx:], argKeyEnd)
			if keyEndIdx == -1 {
				break
			}
			keyEndIdx += keyStartIdx

			keyName := strings.TrimSpace(toolCallText[keyStartIdx+len(argKeyStart) : keyEndIdx])

			// arg_valueの開始を探す（arg_keyの直後）
			valueStartIdx := keyEndIdx + len(argKeyEnd)
			if !strings.HasPrefix(toolCallText[valueStartIdx:], argValueStart) {
				argSearchPos = valueStartIdx
				continue
			}

			// arg_valueの終了を探す
			valueEndIdx := strings.Index(toolCallText[valueStartIdx:], argValueEnd)
			if valueEndIdx == -1 {
				break
			}
			valueEndIdx += valueStartIdx

			value := strings.TrimSpace(toolCallText[valueStartIdx+len(argValueStart) : valueEndIdx])

			// 最初のarg_keyを関数名として扱う（GLM 4.5の仕様）
			if functionName == "" && keyName != "" {
				functionName = keyName
				// 最初のキーは関数名なので、引数には含めない
			} else if keyName != "" {
				// JSON値としてパース試行
				var jsonValue any
				if err := json.Unmarshal([]byte(value), &jsonValue); err == nil {
					args[keyName] = jsonValue
				} else {
					// JSONでない場合は文字列として扱う
					args[keyName] = value
				}
			}

			argSearchPos = valueEndIdx + len(argValueEnd)
		}

		// 関数名が見つからない場合はスキップ
		if functionName == "" {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// 引数をJSONに変換
		argsBytes, _ := json.Marshal(args)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("GLM 4.5 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = endIdx + len(toolCallEnd)
	}

	return toolCalls
}

// extractQwen3CoderXMLToolCalls は Qwen3-Coder XML 形式のツール呼び出しを抽出
// 形式: <tool_call><function>funcName</function><parameter>key=value</parameter>...</tool_call>
func extractQwen3CoderXMLToolCalls(text string) []ToolCall {
	// Qwen3-Coder XMLのタグ
	const (
		toolCallStart = "<tool_call>"
		toolCallEnd   = "</tool_call>"
		functionStart = "<function>"
		functionEnd   = "</function>"
		paramStart    = "<parameter>"
		paramEnd      = "</parameter>"
	)

	// tool_callタグの検出
	if !strings.Contains(text, toolCallStart) {
		return nil
	}

	var toolCalls []ToolCall

	// 複数のtool_callを抽出
	searchPos := 0
	for {
		// tool_callの開始を探す
		startIdx := strings.Index(text[searchPos:], toolCallStart)
		if startIdx == -1 {
			break
		}
		startIdx += searchPos

		// tool_callの終了を探す
		endIdx := strings.Index(text[startIdx:], toolCallEnd)
		if endIdx == -1 {
			break
		}
		endIdx += startIdx

		// 1つのツール呼び出しブロック
		toolCallText := text[startIdx+len(toolCallStart) : endIdx]

		// 関数名の抽出
		funcStartIdx := strings.Index(toolCallText, functionStart)
		if funcStartIdx == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		funcEndIdx := strings.Index(toolCallText[funcStartIdx:], functionEnd)
		if funcEndIdx == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}
		funcEndIdx += funcStartIdx

		functionName := strings.TrimSpace(toolCallText[funcStartIdx+len(functionStart) : funcEndIdx])

		// 空の関数名はスキップ
		if functionName == "" {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// パラメータの抽出
		args := make(map[string]any)

		paramSearchPos := funcEndIdx + len(functionEnd)
		for {
			// parameterの開始を探す
			pStartIdx := strings.Index(toolCallText[paramSearchPos:], paramStart)
			if pStartIdx == -1 {
				break
			}
			pStartIdx += paramSearchPos

			// parameterの終了を探す
			pEndIdx := strings.Index(toolCallText[pStartIdx:], paramEnd)
			if pEndIdx == -1 {
				break
			}
			pEndIdx += pStartIdx

			paramText := strings.TrimSpace(toolCallText[pStartIdx+len(paramStart) : pEndIdx])

			// key=value形式をパース
			eqIdx := strings.Index(paramText, "=")
			if eqIdx != -1 {
				key := strings.TrimSpace(paramText[:eqIdx])
				value := strings.TrimSpace(paramText[eqIdx+1:])

				if key != "" {
					// JSON値としてパース試行
					var jsonValue any
					if err := json.Unmarshal([]byte(value), &jsonValue); err == nil {
						args[key] = jsonValue
					} else {
						// JSONでない場合は文字列として扱う
						args[key] = value
					}
				}
			}

			paramSearchPos = pEndIdx + len(paramEnd)
		}

		// 引数をJSONに変換
		argsBytes, _ := json.Marshal(args)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Qwen3-Coder XML Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = endIdx + len(toolCallEnd)
	}

	return toolCalls
}

// extractXiaomiMiMoToolCalls は Xiaomi MiMo 形式のツール呼び出しを抽出
// 形式: <tool_call>name=functionName, arguments={JSON}</tool_call>
func extractXiaomiMiMoToolCalls(text string) []ToolCall {
	// Xiaomi MiMoのタグ
	const (
		toolCallStart = "<tool_call>"
		toolCallEnd   = "</tool_call>"
	)

	// tool_callタグの検出
	if !strings.Contains(text, toolCallStart) {
		return nil
	}

	var toolCalls []ToolCall

	// 複数のtool_callを抽出
	searchPos := 0
	for {
		// tool_callの開始を探す
		startIdx := strings.Index(text[searchPos:], toolCallStart)
		if startIdx == -1 {
			break
		}
		startIdx += searchPos

		// tool_callの終了を探す
		endIdx := strings.Index(text[startIdx:], toolCallEnd)
		if endIdx == -1 {
			break
		}
		endIdx += startIdx

		// 1つのツール呼び出しブロック
		toolCallText := strings.TrimSpace(text[startIdx+len(toolCallStart) : endIdx])

		// "name=" で始まることを確認
		if !strings.HasPrefix(toolCallText, "name=") {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// ", arguments=" で分割
		parts := strings.SplitN(toolCallText, ", arguments=", 2)
		if len(parts) != 2 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// 関数名の抽出（"name=" を除去）
		functionName := strings.TrimSpace(strings.TrimPrefix(parts[0], "name="))

		// 空の関数名はスキップ
		if functionName == "" {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// 引数の抽出
		argsText := strings.TrimSpace(parts[1])

		// JSON引数のパース
		var argsMap map[string]any
		if argsText != "" {
			if err := json.Unmarshal([]byte(argsText), &argsMap); err != nil {
				searchPos = endIdx + len(toolCallEnd)
				continue
			}
		} else {
			argsMap = make(map[string]any)
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Xiaomi MiMo Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = endIdx + len(toolCallEnd)
	}

	return toolCalls
}

// extractSeedOSSToolCalls は Seed-OSS 形式のツール呼び出しを抽出
// 形式: <seed:tool_call><function>funcName</function><parameter>key=value</parameter>...</seed:tool_call>
func extractSeedOSSToolCalls(text string) []ToolCall {
	// Seed-OSSのタグ
	const (
		toolCallStart = "<seed:tool_call>"
		toolCallEnd   = "</seed:tool_call>"
		functionStart = "<function>"
		functionEnd   = "</function>"
		paramStart    = "<parameter>"
		paramEnd      = "</parameter>"
	)

	// seed:tool_callタグの検出
	if !strings.Contains(text, toolCallStart) {
		return nil
	}

	var toolCalls []ToolCall

	// 複数のseed:tool_callを抽出
	searchPos := 0
	for {
		// seed:tool_callの開始を探す
		startIdx := strings.Index(text[searchPos:], toolCallStart)
		if startIdx == -1 {
			break
		}
		startIdx += searchPos

		// seed:tool_callの終了を探す
		endIdx := strings.Index(text[startIdx:], toolCallEnd)
		if endIdx == -1 {
			break
		}
		endIdx += startIdx

		// 1つのツール呼び出しブロック
		toolCallText := text[startIdx+len(toolCallStart) : endIdx]

		// 関数名の抽出
		funcStartIdx := strings.Index(toolCallText, functionStart)
		if funcStartIdx == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		funcEndIdx := strings.Index(toolCallText[funcStartIdx:], functionEnd)
		if funcEndIdx == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}
		funcEndIdx += funcStartIdx

		functionName := strings.TrimSpace(toolCallText[funcStartIdx+len(functionStart) : funcEndIdx])

		// 空の関数名はスキップ
		if functionName == "" {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// パラメータの抽出
		args := make(map[string]any)

		paramSearchPos := funcEndIdx + len(functionEnd)
		for {
			// parameterの開始を探す
			pStartIdx := strings.Index(toolCallText[paramSearchPos:], paramStart)
			if pStartIdx == -1 {
				break
			}
			pStartIdx += paramSearchPos

			// parameterの終了を探す
			pEndIdx := strings.Index(toolCallText[pStartIdx:], paramEnd)
			if pEndIdx == -1 {
				break
			}
			pEndIdx += pStartIdx

			paramText := strings.TrimSpace(toolCallText[pStartIdx+len(paramStart) : pEndIdx])

			// key=value形式をパース
			eqIdx := strings.Index(paramText, "=")
			if eqIdx != -1 {
				key := strings.TrimSpace(paramText[:eqIdx])
				value := strings.TrimSpace(paramText[eqIdx+1:])

				if key != "" {
					// JSON値としてパース試行
					var jsonValue any
					if err := json.Unmarshal([]byte(value), &jsonValue); err == nil {
						args[key] = jsonValue
					} else {
						// JSONでない場合は文字列として扱う
						args[key] = value
					}
				}
			}

			paramSearchPos = pEndIdx + len(paramEnd)
		}

		// 引数をJSONに変換
		argsBytes, _ := json.Marshal(args)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Seed-OSS Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = endIdx + len(toolCallEnd)
	}

	return toolCalls
}

// extractNemotronV2ToolCalls は Nemotron v2 形式のツール呼び出しを抽出
// 形式: <TOOLCALL>[{"name": "func", "arguments": {...}}]</TOOLCALL>
func extractNemotronV2ToolCalls(text string) []ToolCall {
	// Nemotron v2のタグ
	const (
		toolCallStart = "<TOOLCALL>"
		toolCallEnd   = "</TOOLCALL>"
	)

	// TOOLCALLタグの検出
	startIdx := strings.Index(text, toolCallStart)
	if startIdx == -1 {
		return nil
	}

	// TOOLCALLの終了を探す
	endIdx := strings.Index(text[startIdx:], toolCallEnd)
	var toolCallText string
	if endIdx == -1 {
		toolCallText = text[startIdx+len(toolCallStart):]
	} else {
		toolCallText = text[startIdx+len(toolCallStart) : startIdx+endIdx]
	}

	toolCallText = strings.TrimSpace(toolCallText)

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(toolCallText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// nameの取得
		functionName, ok := tcData["name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// argumentsの取得
		var argsBytes []byte
		if args, exists := tcData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Nemotron v2 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractApertusToolCalls は Apertus 形式のツール呼び出しを抽出
// 形式: <|tools_prefix|>[{"functionName": {arguments}}]<|tools_suffix|>
func extractApertusToolCalls(text string) []ToolCall {
	// Apertusのタグ
	const (
		toolsPrefix = "<|tools_prefix|>"
		toolsSuffix = "<|tools_suffix|>"
	)

	// tools_prefixタグの検出
	startIdx := strings.Index(text, toolsPrefix)
	if startIdx == -1 {
		return nil
	}

	// tools_suffixの終了を探す
	endIdx := strings.Index(text[startIdx:], toolsSuffix)
	var toolsText string
	if endIdx == -1 {
		toolsText = text[startIdx+len(toolsPrefix):]
	} else {
		toolsText = text[startIdx+len(toolsPrefix) : startIdx+endIdx]
	}

	toolsText = strings.TrimSpace(toolsText)

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(toolsText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// 各オブジェクトの最初のキーを関数名として扱う
		for functionName, args := range tcData {
			if functionName == "" {
				continue
			}

			// 引数の処理
			var argsBytes []byte
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}

			toolCalls = append(toolCalls, ToolCall{
				ID:       generateToolCallID(),
				Type:     "function",
				Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
			})

			logDebug("Apertus Tool Call Detected", map[string]any{
				"Function": functionName,
				"Args":     string(argsBytes),
			})

			// Apertus形式では各オブジェクトに1つのキーのみ
			break
		}
	}

	return toolCalls
}

// extractLFM2ToolCalls は LFM2 形式のツール呼び出しを抽出
// 形式: <|tool_call_start|>[{"name": "func", "arguments": {...}}]<|tool_call_end|>
func extractLFM2ToolCalls(text string) []ToolCall {
	// LFM2のタグ
	const (
		toolCallStart = "<|tool_call_start|>"
		toolCallEnd   = "<|tool_call_end|>"
	)

	// tool_call_startタグの検出
	startIdx := strings.Index(text, toolCallStart)
	if startIdx == -1 {
		return nil
	}

	// tool_call_endの終了を探す
	endIdx := strings.Index(text[startIdx:], toolCallEnd)
	var toolCallText string
	if endIdx == -1 {
		toolCallText = text[startIdx+len(toolCallStart):]
	} else {
		toolCallText = text[startIdx+len(toolCallStart) : startIdx+endIdx]
	}

	toolCallText = strings.TrimSpace(toolCallText)

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(toolCallText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// nameの取得
		functionName, ok := tcData["name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// argumentsの取得
		var argsBytes []byte
		if args, exists := tcData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("LFM2 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractMiniMaxM2ToolCalls は MiniMax-M2 形式のツール呼び出しを抽出
// 形式: <minimax:tool_call><invoke name="func"><parameter name="key">value</parameter>...</invoke></minimax:tool_call>
func extractMiniMaxM2ToolCalls(text string) []ToolCall {
	// MiniMax-M2のタグ
	const (
		toolCallStart = "<minimax:tool_call>"
		toolCallEnd   = "</minimax:tool_call>"
		invokeStart   = "<invoke name="
		invokeEnd     = "</invoke>"
		paramStart    = "<parameter name="
		paramEnd      = "</parameter>"
	)

	// minimax:tool_callタグの検出
	if !strings.Contains(text, toolCallStart) {
		return nil
	}

	var toolCalls []ToolCall

	// 複数のminimax:tool_callを抽出
	searchPos := 0
	for {
		// minimax:tool_callの開始を探す
		startIdx := strings.Index(text[searchPos:], toolCallStart)
		if startIdx == -1 {
			break
		}
		startIdx += searchPos

		// minimax:tool_callの終了を探す
		endIdx := strings.Index(text[startIdx:], toolCallEnd)
		if endIdx == -1 {
			break
		}
		endIdx += startIdx

		// 1つのツール呼び出しブロック
		toolCallText := text[startIdx+len(toolCallStart) : endIdx]

		// invokeタグから関数名を抽出
		invokeIdx := strings.Index(toolCallText, invokeStart)
		if invokeIdx == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// 関数名の抽出（引用符で囲まれている）
		nameStart := invokeIdx + len(invokeStart)
		nameEnd := strings.Index(toolCallText[nameStart:], `"`)
		if nameEnd == -1 {
			// 単一引用符の場合も試す
			nameEnd = strings.Index(toolCallText[nameStart:], `'`)
		}
		if nameEnd == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}
		nameEnd += nameStart

		functionName := strings.TrimSpace(toolCallText[nameStart:nameEnd])

		// 空の関数名はスキップ
		if functionName == "" {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}

		// パラメータの抽出
		args := make(map[string]any)

		// invokeの終了位置を探す
		invokeEndIdx := strings.Index(toolCallText[nameEnd:], invokeEnd)
		if invokeEndIdx == -1 {
			searchPos = endIdx + len(toolCallEnd)
			continue
		}
		invokeEndIdx += nameEnd

		invokeContent := toolCallText[nameEnd:invokeEndIdx]

		paramSearchPos := 0
		for {
			// parameterの開始を探す
			pStartIdx := strings.Index(invokeContent[paramSearchPos:], paramStart)
			if pStartIdx == -1 {
				break
			}
			pStartIdx += paramSearchPos

			// パラメータ名の抽出（引用符で囲まれている）
			pNameStart := pStartIdx + len(paramStart)
			pNameEnd := strings.Index(invokeContent[pNameStart:], `"`)
			if pNameEnd == -1 {
				pNameEnd = strings.Index(invokeContent[pNameStart:], `'`)
			}
			if pNameEnd == -1 {
				break
			}
			pNameEnd += pNameStart

			paramName := strings.TrimSpace(invokeContent[pNameStart:pNameEnd])

			// 値の開始（">" の後）
			valueStart := strings.Index(invokeContent[pNameEnd:], ">")
			if valueStart == -1 {
				break
			}
			valueStart += pNameEnd + 1

			// parameterの終了を探す
			pEndIdx := strings.Index(invokeContent[valueStart:], paramEnd)
			if pEndIdx == -1 {
				break
			}
			pEndIdx += valueStart

			paramValue := strings.TrimSpace(invokeContent[valueStart:pEndIdx])

			if paramName != "" {
				// JSON値としてパース試行
				var jsonValue any
				if err := json.Unmarshal([]byte(paramValue), &jsonValue); err == nil {
					args[paramName] = jsonValue
				} else {
					// JSONでない場合は文字列として扱う
					args[paramName] = paramValue
				}
			}

			paramSearchPos = pEndIdx + len(paramEnd)
		}

		// 引数をJSONに変換
		argsBytes, _ := json.Marshal(args)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("MiniMax-M2 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = endIdx + len(toolCallEnd)
	}

	return toolCalls
}

// extractKimiK2ToolCalls は Kimi K2 形式のツール呼び出しを抽出
// 形式: <|tool_calls_section_begin|><|tool_call_begin|>functionName<|tool_call_argument_begin|>{JSON}<|tool_call_end|><|tool_calls_section_end|>
func extractKimiK2ToolCalls(text string) []ToolCall {
	// Kimi K2のタグ
	const (
		sectionBegin  = "<|tool_calls_section_begin|>"
		sectionEnd    = "<|tool_calls_section_end|>"
		toolCallBegin = "<|tool_call_begin|>"
		toolCallEnd   = "<|tool_call_end|>"
		argumentBegin = "<|tool_call_argument_begin|>"
	)

	// tool_calls_section_beginタグの検出
	if !strings.Contains(text, sectionBegin) {
		return nil
	}

	// セクション全体を抽出
	startIdx := strings.Index(text, sectionBegin)
	if startIdx == -1 {
		return nil
	}

	endIdx := strings.Index(text[startIdx:], sectionEnd)
	var sectionText string
	if endIdx == -1 {
		sectionText = text[startIdx+len(sectionBegin):]
	} else {
		sectionText = text[startIdx+len(sectionBegin) : startIdx+endIdx]
	}

	var toolCalls []ToolCall

	// 複数のtool_callを抽出
	searchPos := 0
	for {
		// tool_call_beginを探す
		beginIdx := strings.Index(sectionText[searchPos:], toolCallBegin)
		if beginIdx == -1 {
			break
		}
		beginIdx += searchPos

		// 関数名の開始位置
		nameStart := beginIdx + len(toolCallBegin)

		// tool_call_argument_beginを探す
		argBeginIdx := strings.Index(sectionText[nameStart:], argumentBegin)
		if argBeginIdx == -1 {
			break
		}
		argBeginIdx += nameStart

		// 関数名の抽出
		functionName := strings.TrimSpace(sectionText[nameStart:argBeginIdx])

		// 空の関数名はスキップ
		if functionName == "" {
			searchPos = argBeginIdx + len(argumentBegin)
			continue
		}

		// 引数の開始位置
		argsStart := argBeginIdx + len(argumentBegin)

		// tool_call_endを探す
		endIdx := strings.Index(sectionText[argsStart:], toolCallEnd)
		if endIdx == -1 {
			break
		}
		endIdx += argsStart

		// 引数テキストの抽出
		argsText := strings.TrimSpace(sectionText[argsStart:endIdx])

		// JSON引数のパース
		var argsMap map[string]any
		if argsText != "" {
			if err := json.Unmarshal([]byte(argsText), &argsMap); err != nil {
				searchPos = endIdx + len(toolCallEnd)
				continue
			}
		} else {
			argsMap = make(map[string]any)
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Kimi K2 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = endIdx + len(toolCallEnd)
	}

	return toolCalls
}

// extractApriel15ToolCalls は Apriel 1.5 形式のツール呼び出しを抽出
// 形式: <tool_calls><name>func</name>, <arguments>{JSON}</arguments></tool_calls>
func extractApriel15ToolCalls(text string) []ToolCall {
	// Apriel 1.5のタグ
	const (
		toolCallsStart = "<tool_calls>"
		toolCallsEnd   = "</tool_calls>"
		nameStart      = "<name>"
		nameEnd        = "</name>"
		argumentsStart = "<arguments>"
		argumentsEnd   = "</arguments>"
	)

	// tool_callsタグの検出
	if !strings.Contains(text, toolCallsStart) {
		return nil
	}

	// tool_callsセクション全体を抽出
	startIdx := strings.Index(text, toolCallsStart)
	if startIdx == -1 {
		return nil
	}

	endIdx := strings.Index(text[startIdx:], toolCallsEnd)
	var toolCallsText string
	if endIdx == -1 {
		toolCallsText = text[startIdx+len(toolCallsStart):]
	} else {
		toolCallsText = text[startIdx+len(toolCallsStart) : startIdx+endIdx]
	}

	var toolCalls []ToolCall

	// 複数のname/argumentsペアを抽出
	searchPos := 0
	for {
		// nameタグの開始を探す
		nStartIdx := strings.Index(toolCallsText[searchPos:], nameStart)
		if nStartIdx == -1 {
			break
		}
		nStartIdx += searchPos

		// nameタグの終了を探す
		nEndIdx := strings.Index(toolCallsText[nStartIdx:], nameEnd)
		if nEndIdx == -1 {
			break
		}
		nEndIdx += nStartIdx

		// 関数名の抽出
		functionName := strings.TrimSpace(toolCallsText[nStartIdx+len(nameStart) : nEndIdx])

		// 空の関数名はスキップ
		if functionName == "" {
			searchPos = nEndIdx + len(nameEnd)
			continue
		}

		// argumentsタグの開始を探す（nameの後）
		aStartIdx := strings.Index(toolCallsText[nEndIdx:], argumentsStart)
		if aStartIdx == -1 {
			searchPos = nEndIdx + len(nameEnd)
			continue
		}
		aStartIdx += nEndIdx

		// argumentsタグの終了を探す
		aEndIdx := strings.Index(toolCallsText[aStartIdx:], argumentsEnd)
		if aEndIdx == -1 {
			break
		}
		aEndIdx += aStartIdx

		// 引数テキストの抽出
		argsText := strings.TrimSpace(toolCallsText[aStartIdx+len(argumentsStart) : aEndIdx])

		// JSON引数のパース
		var argsMap map[string]any
		if argsText != "" {
			if err := json.Unmarshal([]byte(argsText), &argsMap); err != nil {
				searchPos = aEndIdx + len(argumentsEnd)
				continue
			}
		} else {
			argsMap = make(map[string]any)
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Apriel 1.5 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})

		searchPos = aEndIdx + len(argumentsEnd)
	}

	return toolCalls
}

// extractFirefunctionV2ToolCalls は Firefunction v2 形式のツール呼び出しを抽出
// 形式:  functools[{"name": "func", "arguments": {...}}]
func extractFirefunctionV2ToolCalls(text string) []ToolCall {
	// Firefunction v2のプレフィックス
	const prefix = " functools"

	// プレフィックスの検出
	prefixIdx := strings.Index(text, prefix)
	if prefixIdx == -1 {
		return nil
	}

	// JSON配列の開始位置（プレフィックスの直後）
	jsonStart := prefixIdx + len(prefix)
	jsonText := strings.TrimSpace(text[jsonStart:])

	// JSON配列の開始を確認
	if !strings.HasPrefix(jsonText, "[") {
		return nil
	}

	// JSON配列の終了を探す
	jsonEnd := strings.LastIndex(jsonText, "]")
	if jsonEnd == -1 {
		return nil
	}

	jsonText = jsonText[:jsonEnd+1]

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(jsonText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// nameの取得
		functionName, ok := tcData["name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// argumentsの取得
		var argsBytes []byte
		if args, exists := tcData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Firefunction v2 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractFunctionaryV31Llama31ToolCalls は Functionary v3.1 Llama 3.1 形式のツール呼び出しを抽出
// 形式: <function=functionName>{JSON}</function>
func extractFunctionaryV31Llama31ToolCalls(text string) []ToolCall {
	// Functionary v3.1 Llama 3.1のタグパターン
	// <function=functionName> ... </function>
	closeTag := `</function>`
	matches := regexFunctionaryV31Llama31.FindAllStringSubmatchIndex(text, -1)

	if len(matches) == 0 {
		return nil
	}

	var toolCalls []ToolCall

	for _, match := range matches {
		// match[0], match[1]: 全体マッチ（<function=functionName>）
		// match[2], match[3]: 関数名のキャプチャグループ

		if len(match) < 4 {
			continue
		}

		functionName := strings.TrimSpace(text[match[2]:match[3]])

		// 空の関数名はスキップ
		if functionName == "" {
			continue
		}

		// JSON引数の抽出（<function=...>の後から</function>まで）
		jsonStart := match[1] // <function=...>の終了位置
		remainingText := text[jsonStart:]

		closeIdx := strings.Index(remainingText, closeTag)
		if closeIdx == -1 {
			continue
		}

		jsonText := strings.TrimSpace(remainingText[:closeIdx])

		// JSON引数のパース
		var argsMap map[string]any
		if jsonText != "" {
			if err := json.Unmarshal([]byte(jsonText), &argsMap); err != nil {
				continue
			}
		} else {
			argsMap = make(map[string]any)
		}

		argsBytes, _ := json.Marshal(argsMap)

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Functionary v3.1 Llama 3.1 Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractLlama3XToolCalls は Llama 3.x 形式のツール呼び出しを抽出
// 形式: {"type": "function", "name": "functionName", "parameters": {...}}
func extractLlama3XToolCalls(text string) []ToolCall {
	// Llama 3.xのJSON形式パターン
	// {"type": "function", "name": "...", "parameters": {...}}
	matches := regexLlama3X.FindAllStringSubmatchIndex(text, -1)

	if len(matches) == 0 {
		return nil
	}

	var toolCalls []ToolCall

	for _, match := range matches {
		// match[0], match[1]: 全体マッチ
		// match[2], match[3]: 関数名のキャプチャグループ

		if len(match) < 4 {
			continue
		}

		functionName := strings.TrimSpace(text[match[2]:match[3]])

		// 空の関数名はスキップ
		if functionName == "" {
			continue
		}

		// parametersの値を抽出（match[1]の位置から）
		jsonStart := match[1]
		remainingText := text[jsonStart:]

		// parametersのJSONオブジェクトを抽出
		// 中括弧のバランスを取りながら抽出
		braceCount := 0
		jsonEnd := -1
		inString := false
		escape := false

		for i, ch := range remainingText {
			if escape {
				escape = false
				continue
			}

			if ch == '\\' {
				escape = true
				continue
			}

			if ch == '"' {
				inString = !inString
				continue
			}

			if !inString {
				if ch == '{' {
					braceCount++
				} else if ch == '}' {
					braceCount--
					if braceCount == 0 {
						jsonEnd = i + 1
						break
					}
				}
			}
		}

		if jsonEnd == -1 {
			continue
		}

		// 完全なJSONオブジェクトを抽出
		fullJsonText := remainingText[:jsonEnd]

		// JSON全体をパース
		var fullObj map[string]any
		if err := json.Unmarshal([]byte(fullJsonText), &fullObj); err != nil {
			continue
		}

		// parametersを取得
		var argsBytes []byte
		if params, exists := fullObj["parameters"]; exists {
			if paramsMap, ok := params.(map[string]any); ok {
				argsBytes, _ = json.Marshal(paramsMap)
			} else {
				argsBytes = []byte("{}")
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Llama 3.x Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractMagistralToolCalls は Magistral 形式のツール呼び出しを抽出
// 形式: [TOOLCALLS][{"name": "func", "arguments": {...}}]
func extractMagistralToolCalls(text string) []ToolCall {
	// Magistralのプレフィックス
	const prefix = "[TOOLCALLS]"

	// プレフィックスの検出
	prefixIdx := strings.Index(text, prefix)
	if prefixIdx == -1 {
		return nil
	}

	// JSON配列の開始位置（プレフィックスの直後）
	jsonStart := prefixIdx + len(prefix)
	jsonText := strings.TrimSpace(text[jsonStart:])

	// JSON配列の開始を確認
	if !strings.HasPrefix(jsonText, "[") {
		return nil
	}

	// JSON配列の終了を探す
	jsonEnd := strings.LastIndex(jsonText, "]")
	if jsonEnd == -1 {
		return nil
	}

	jsonText = jsonText[:jsonEnd+1]

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(jsonText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// nameの取得
		functionName, ok := tcData["name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// argumentsの取得
		var argsBytes []byte
		if args, exists := tcData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       generateToolCallID(),
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Magistral Tool Call Detected", map[string]any{
			"Function": functionName,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractMistralNemoToolCalls は Mistral Nemo 形式のツール呼び出しを抽出
// 形式: [TOOL_CALLS][{"name": "func", "arguments": {...}, "id": "123456789"}]
func extractMistralNemoToolCalls(text string) []ToolCall {
	// Mistral Nemoのプレフィックス
	const prefix = "[TOOL_CALLS]"

	// プレフィックスの検出
	prefixIdx := strings.Index(text, prefix)
	if prefixIdx == -1 {
		return nil
	}

	// JSON配列の開始位置（プレフィックスの直後）
	jsonStart := prefixIdx + len(prefix)
	jsonText := strings.TrimSpace(text[jsonStart:])

	// JSON配列の開始を確認
	if !strings.HasPrefix(jsonText, "[") {
		return nil
	}

	// JSON配列の終了を探す
	jsonEnd := strings.LastIndex(jsonText, "]")
	if jsonEnd == -1 {
		return nil
	}

	jsonText = jsonText[:jsonEnd+1]

	// JSON配列のパース
	var toolCallsData []map[string]any
	if err := json.Unmarshal([]byte(jsonText), &toolCallsData); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	for _, tcData := range toolCallsData {
		// nameの取得
		functionName, ok := tcData["name"].(string)
		if !ok || functionName == "" {
			continue
		}

		// idの取得（Mistral Nemo特有）
		toolCallID := ""
		if id, ok := tcData["id"].(string); ok {
			toolCallID = id
		}

		// IDが指定されていない場合は生成
		if toolCallID == "" {
			toolCallID = generateToolCallID()
		}

		// argumentsの取得
		var argsBytes []byte
		if args, exists := tcData["arguments"]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					argsBytes = []byte("{}")
				}
			} else {
				argsBytes, _ = json.Marshal(args)
			}
		} else {
			argsBytes = []byte("{}")
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:       toolCallID,
			Type:     "function",
			Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
		})

		logDebug("Mistral Nemo Tool Call Detected", map[string]any{
			"Function": functionName,
			"ID":       toolCallID,
			"Args":     string(argsBytes),
		})
	}

	return toolCalls
}

// extractGenericToolCalls は汎用的なJSONベースのツール呼び出しを抽出
// 様々なフォーマットのJSONから toolcalls/toolcall フィールドを探索
func extractGenericToolCalls(text string) []ToolCall {
	// JSON全体をパース試行
	text = strings.TrimSpace(text)

	// JSONブロックの抽出（中括弧で始まる部分）
	jsonStart := strings.Index(text, "{")
	if jsonStart == -1 {
		return nil
	}

	// 最後の閉じ括弧を探す
	jsonEnd := strings.LastIndex(text, "}")
	if jsonEnd == -1 || jsonEnd <= jsonStart {
		return nil
	}

	jsonStr := text[jsonStart : jsonEnd+1]

	var data map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
		return nil
	}

	var toolCalls []ToolCall

	// パターン1: "toolcalls" 配列フィールド
	if toolCallsArray, ok := data["toolcalls"].([]any); ok {
		for _, tc := range toolCallsArray {
			if tcMap, ok := tc.(map[string]any); ok {
				if toolCall := parseGenericToolCallObject(tcMap); toolCall != nil {
					toolCalls = append(toolCalls, *toolCall)
				}
			}
		}
		if len(toolCalls) > 0 {
			logDebug("Generic Tool Calls Detected", map[string]any{
				"Pattern": "toolcalls array",
				"Count":   len(toolCalls),
			})
			return toolCalls
		}
	}

	// パターン2: "tool_calls" 配列フィールド（アンダースコア付き）
	if toolCallsArray, ok := data["tool_calls"].([]any); ok {
		for _, tc := range toolCallsArray {
			if tcMap, ok := tc.(map[string]any); ok {
				if toolCall := parseGenericToolCallObject(tcMap); toolCall != nil {
					toolCalls = append(toolCalls, *toolCall)
				}
			}
		}
		if len(toolCalls) > 0 {
			logDebug("Generic Tool Calls Detected", map[string]any{
				"Pattern": "tool_calls array",
				"Count":   len(toolCalls),
			})
			return toolCalls
		}
	}

	// パターン3: "toolcall" 単一オブジェクト
	if toolCallObj, ok := data["toolcall"].(map[string]any); ok {
		if toolCall := parseGenericToolCallObject(toolCallObj); toolCall != nil {
			logDebug("Generic Tool Call Detected", map[string]any{
				"Pattern":  "toolcall object",
				"Function": toolCall.Function.Name,
			})
			return []ToolCall{*toolCall}
		}
	}

	// パターン4: "tool_call" 単一オブジェクト（アンダースコア付き）
	if toolCallObj, ok := data["tool_call"].(map[string]any); ok {
		if toolCall := parseGenericToolCallObject(toolCallObj); toolCall != nil {
			logDebug("Generic Tool Call Detected", map[string]any{
				"Pattern":  "tool_call object",
				"Function": toolCall.Function.Name,
			})
			return []ToolCall{*toolCall}
		}
	}

	// パターン5: "response" フィールド（llama.cpp互換）
	if response, ok := data["response"]; ok {
		// responseフィールドがある場合、これはコンテンツであってツール呼び出しではない
		// TCGWはツール呼び出し抽出専用なので、nilを返す
		logDebug("Generic Parser: response field detected (not a tool call)", map[string]any{
			"Response": response,
		})
		return nil
	}

	return nil
}

// parseGenericToolCallObject は汎用的なツール呼び出しオブジェクトをパース
func parseGenericToolCallObject(obj map[string]any) *ToolCall {
	// 関数名の取得（複数のフィールド名に対応）
	var functionName string
	for _, key := range []string{"name", "function", "function_name", "tool", "tool_name"} {
		if name, ok := obj[key].(string); ok && name != "" {
			functionName = name
			break
		}
	}

	if functionName == "" {
		return nil
	}

	// 引数の取得（複数のフィールド名に対応）
	var argsBytes []byte
	for _, key := range []string{"arguments", "args", "parameters", "params", "input"} {
		if args, exists := obj[key]; exists {
			if argsMap, ok := args.(map[string]any); ok {
				argsBytes, _ = json.Marshal(argsMap)
				break
			} else if argsStr, ok := args.(string); ok {
				// 文字列の場合、JSONとしてパース試行
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(argsStr), &argsMap); err == nil {
					argsBytes = []byte(argsStr)
				} else {
					// JSONでない場合は空オブジェクト
					argsBytes = []byte("{}")
				}
				break
			}
		}
	}

	if argsBytes == nil {
		argsBytes = []byte("{}")
	}

	return &ToolCall{
		ID:       generateToolCallID(),
		Type:     "function",
		Function: ToolCallFunction{Name: functionName, Arguments: string(argsBytes)},
	}
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
		logDebug("Bifrost Response Error", backendResp)
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

	// サーバー起動
	if err := emulateRouter.Run(emulatePort); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start server: %v\n", err)
		os.Exit(1)
	}
}
