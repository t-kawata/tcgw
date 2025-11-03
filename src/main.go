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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
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
	TOOL_SYSTEM_PROMPT = `You are a helpful AI assistant with access to the following tools:

<tools>
{{TOOLS_XML}}
</tools>

When you need to use a tool, respond in this exact format:
<function_calls>
  <invoke name="tool_name">
    <parameter name="param_name">value</parameter>
  </invoke>
</function_calls>

You can call multiple tools by adding more <invoke> blocks.
Always use the exact tool names and parameter names as specified.`

	FUNCTION_CALLS_PATTERN = `<function_calls>([\\s\\S]*?)</function_calls>`
	INVOKE_PATTERN         = `<invoke\\s+name="([^"]+)">([\\s\\S]*?)</invoke>`
	// [^<]* から ([\\s\\S]*?) に変更。値にエスケープされていない < が含まれる等のエッジケースに対応
	PARAMETER_PATTERN = `<parameter\\s+name="([^"]+)">([\\s\\S]*?)</parameter>`
	JSON_PATTERN      = `\\{[^{}]*"tool_calls"[^{}]*\\}`
	// (s?)フラグを追加して複数行のマッチングに対応
	MARKDOWN_JSON_PATTERN = "(?s)```(?:json)?\\\\s*([^`]+)```"
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
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}
type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []Tool    `json:"tools,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Temperature *float32  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	TopP        *float32  `json:"top_p,omitempty"`
}

// --- 型定義 (レスポンス) ---
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}
type ResponseMessage struct {
	Role      string     `json:"role"`
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
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
		b[i] = charset[rand.Intn(len(charset))]
	}
	return "call_" + string(b)
}
func generateResponseID() string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
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
func embedToolsIntoPrompt(req *ChatCompletionRequest) {
	if len(req.Tools) == 0 {
		return
	}
	toolsXML := generateToolsXML(req.Tools)
	systemPrompt := strings.ReplaceAll(TOOL_SYSTEM_PROMPT, "{{TOOLS_XML}}", toolsXML)
	if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
		req.Messages[0].Content = systemPrompt + "\n\n" + req.Messages[0].Content
	} else {
		newMessages := make([]Message, 1+len(req.Messages))
		newMessages[0] = Message{Role: "system", Content: systemPrompt}
		copy(newMessages[1:], req.Messages)
		req.Messages = newMessages
	}
	req.Tools = nil // ツール定義を削除 (Bifrostには送らない)
	logDebug("Embedding Tools", map[string]any{
		"System Prompt Len": len(systemPrompt),
		"Messages Count":    len(req.Messages),
	})
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
		return ""
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return ""
	}
	message, ok := choice["message"].(map[string]any)
	if !ok {
		return ""
	}
	content, ok := message["content"].(string)
	if !ok {
		return ""
	}
	return content
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
		c.JSON(400, ErrorResponse{Error: struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code,omitempty"`
		}{Message: fmt.Sprintf("Invalid JSON: %v", err), Type: "invalid_request_error", Code: "invalid_request"}})
		return
	}

	if req.Stream {
		c.JSON(501, ErrorResponse{Error: struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code,omitempty"`
		}{Message: "Streaming is not currently supported", Type: "invalid_request_error"}})
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
	// choices[0].message の部分だけ上書き
	choices, ok := backendResp["choices"].([]any)
	if !ok || len(choices) == 0 {
		// 構造が不正な場合は既存の buildOpenAIResponse にフォールバック
		return nil
	}

	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}

	message, ok := choice["message"].(map[string]any)
	if !ok {
		message = map[string]any{}
		choice["message"] = message
	}

	// ツール呼び出しの有無で分岐
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		message["content"] = nil
		choice["finish_reason"] = "tool_calls"
	} else {
		// contentはそのまま保持
		message["tool_calls"] = []ToolCall{}
		choice["finish_reason"] = "stop"
	}

	// choices配列を更新
	backendResp["choices"] = []any{choice}

	return backendResp
}

// パススルーモード: リクエストをそのままBifrostに転送（ネイティブTool Calling使用）
func handleChatCompletionsPassthrough(c *gin.Context) {
	var req ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, ErrorResponse{Error: struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code,omitempty"`
		}{Message: fmt.Sprintf("Invalid JSON: %v", err), Type: "invalid_request_error", Code: "invalid_request"}})
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

	logDebug("Response Forwarded (Passthrough Mode)", map[string]any{
		"Status": "Success",
	})

	// Bifrostからのレスポンスをそのまま返却
	c.JSON(200, backendResp)
}

func handleHealthCheck(c *gin.Context) {
	c.JSON(200, gin.H{
		"status":    "ok",
		"service":   "tcgw",
		"version":   config.VERSION,
		"mode":      "dual-port",
		"timestamp": time.Now().Unix(),
	})
}

// --- サーバ起動 ---
func main() {
	// グローバルなrandのシード。Go 1.20+では自動シードされるため不要
	// rand.Seed(time.Now().UnixNano())
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
