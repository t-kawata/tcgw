# TCGW (Tool Calling Gateway)

TCGWは、Tool Calling機能を持たない任意のLLMに対して、OpenAI互換のTool Calling APIを提供するGo製のプロキシサーバーです。クライアントからはOpenAI APIと同じインターフェースでアクセスでき、バックエンドのLLMとの間でツール定義の変換とツール呼び出しの抽出を自動的に行います。

## 概要

TCGWは、LLMの応答テキストからXML形式のツール呼び出しを自動的に抽出し、OpenAI互換の形式に変換します。これにより、Llama、Mistral、Gemmaなどのオープンソースモデルでも、OpenAI SDKを使用したツール呼び出しが可能になります。

**デュアルポートモード**に対応しました。エミュレートモードとパススルーモードを同時に提供し、ポートを切り替えるだけでツール呼び出しの動作モードを選択できます。

### 主な機能

- **デュアルポートモード**: エミュレートモードとパススルーモードを同時起動
- **エミュレートモード**: Tool Calling非対応LLMにツール呼び出し機能を提供
- **パススルーモード**: プロバイダーのネイティブTool Calling機能をそのまま利用
- **OpenAI互換API**: クライアントからはOpenAI Chat Completions API形式でアクセス可能
- **自動ツール定義埋め込み**: ツール定義をXML形式に変換してシステムプロンプトに自動挿入（エミュレートモード）
- **堅牢なXML解析**: 不完全なXMLや特殊文字を含むパラメータに対応
- **型推定機能**: パラメータ値の型（文字列、数値、真偽値）を自動判定
- **Bifrost統合**: バックエンドプロキシとしてBifrostを使用し、複数のLLMプロバイダーに対応
- **デバッグモード**: 詳細なログ出力で動作確認とトラブルシューティングが可能

## システム要件

- Go 1.25.x 以上
- Bifrost（バックエンドLLMプロキシ）が稼働していること

## インストール

### 1. 依存パッケージのインストール

```bash
go mod tidy
```

### 2. 環境変数の設定

プロジェクトルートに`.env`ファイルを作成し、以下の設定を記述してください：

```bash
# Bifrost接続設定
BIFROST_URL=http://0.0.0.0:7766

# 認証（オプション）
BIFROST_API_KEY=

# エミュレートモード用ポート（Tool Callingをエミュレート）
EMULATE_PORT=3000

# パススルーモード用ポート（ネイティブTool Callingを使用）
PASSTHROUGH_PORT=3001

# タイムアウト設定（ミリ秒）
REQUEST_TIMEOUT=120000

# デバッグモード
DEBUG_MODE=false
```

#### 環境変数の説明

| 変数名 | 説明 | デフォルト値 | 必須 |
|--------|------|-------------|------|
| `BIFROST_URL` | BifrostサーバーのURL | `http://0.0.0.0:7766` | はい |
| `BIFROST_API_KEY` | Bifrost認証用APIキー | なし | いいえ |
| `EMULATE_PORT` | エミュレートモードのポート番号 | `3000` | いいえ |
| `PASSTHROUGH_PORT` | パススルーモードのポート番号 | `3001` | いいえ |
| `REQUEST_TIMEOUT` | バックエンドへのリクエストタイムアウト（ミリ秒） | `120000` | いいえ |
| `DEBUG_MODE` | デバッグログの出力（`true`/`false`） | `false` | いいえ |

**注意**: `EMULATE_PORT`と`PASSTHROUGH_PORT`は異なるポート番号を指定してください。同じポート番号を指定すると起動時にエラーが発生します。

## 起動方法

### 開発環境での起動

```bash
go run main.go
```

### ビルドして実行

```bash
go build -o tcgw main.go
./tcgw
```

起動に成功すると、以下のようなメッセージが表示されます：

```
🌉 TCGW Proxy Server (Dual-Port Mode)
[TCGW] Server Starting
  Emulate Mode:     0.0.0.0:3000 (Tool Calling Emulation)
  Passthrough Mode: 0.0.0.0:3001 (Native Tool Calling)
  BIFROST:          http://0.0.0.0:7766
```

## 動作モードの選択

TCGWは2つの動作モードを提供します。リクエスト先のポートを切り替えるだけで、モードを選択できます。

### エミュレートモード（ポート3000）

Tool Calling機能を持たないLLMに対して、ツール呼び出し機能をエミュレートします。

**動作**：
1. ツール定義をXML形式に変換してシステムプロンプトに埋め込む
2. LLMの応答からXML形式のツール呼び出しを抽出
3. OpenAI互換形式に変換してクライアントに返却

**用途**：Llama、Mistral、Gemmaなど、Tool Calling非対応のオープンソースモデルを使用する場合

### パススルーモード（ポート3001）

リクエストをそのままBifrostに転送し、プロバイダーのネイティブTool Calling機能を使用します。

**動作**：
1. クライアントからのリクエストをそのままBifrostに転送
2. Bifrostからのレスポンスをそのままクライアントに返却
3. ツール定義の変換や抽出処理は行わない

**用途**：GPT-4、Claude、Geminiなど、ネイティブにTool Callingをサポートするモデルを使用する場合

## 使用方法

### 基本的な使い方

TCGWは、OpenAI Chat Completions APIと完全に互換性があります。以下のようにOpenAI SDKを使用してアクセスできます。

#### Python（OpenAI SDK）の例

```python
from openai import OpenAI

# エミュレートモード使用（Tool Callingをエミュレート）
client_emulate = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="dummy"  # 任意の値
)

# パススルーモード使用（ネイティブTool Calling）
client_passthrough = OpenAI(
    base_url="http://localhost:3001/v1",
    api_key="dummy"  # 任意の値
)

# ツール定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定された都市の天気を取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "都市名"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度の単位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# エミュレートモードでリクエスト送信
response = client_emulate.chat.completions.create(
    model="llama3",  # Tool Calling非対応モデル
    messages=[
        {"role": "user", "content": "東京の天気を教えてください"}
    ],
    tools=tools
)

# または、パススルーモードでリクエスト送信
response = client_passthrough.chat.completions.create(
    model="gpt-4",  # Tool Calling対応モデル
    messages=[
        {"role": "user", "content": "東京の天気を教えてください"}
    ],
    tools=tools
)

# レスポンス処理（どちらのモードでも同じ）
if response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls
    for tool_call in tool_calls:
        print(f"ツール: {tool_call.function.name}")
        print(f"引数: {tool_call.function.arguments}")
```

#### curlでのリクエスト例

```bash
# エミュレートモード（ポート3000）
curl -X POST http://localhost:3000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "東京の天気を教えてください"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "指定された都市の天気を取得します",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "都市名"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'

# パススルーモード（ポート3001）
curl -X POST http://localhost:3001/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "東京の天気を教えてください"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "指定された都市の天気を取得します",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "都市名"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

### レスポンス形式

レスポンス形式は、どちらのモードでもOpenAI互換です。

#### ツール呼び出しがある場合

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1698765432,
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xyz789",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"東京\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

#### 通常のテキスト応答の場合

```json
{
  "id": "chatcmpl-def456",
  "object": "chat.completion",
  "created": 1698765432,
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "東京は日本の首都で、温暖な気候です。",
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

### ヘルスチェック

サーバーの状態を確認するには、以下のエンドポイントにアクセスしてください：

```bash
# エミュレートモード
curl http://localhost:3000/health

# パススルーモード
curl http://localhost:3001/health
```

レスポンス例：

```json
{
  "status": "ok",
  "service": "tcgw",
  "version": "1.1.0",
  "mode": "dual-port",
  "timestamp": 1698765432
}
```

## 動作の仕組み

### エミュレートモード

TCGWは以下の流れで動作します：

1. **リクエスト受信**: クライアントからOpenAI形式のリクエストを受信
2. **ツール定義の変換**: `tools`配列をXML形式に変換し、システムプロンプトに埋め込み
3. **バックエンドへの転送**: 変換したリクエストをBifrostに転送
4. **レスポンス受信**: BifrostからLLMの応答を受信
5. **ツール呼び出しの抽出**: 応答テキストからXML形式のツール呼び出しを抽出
6. **レスポンス変換**: 抽出したツール呼び出しをOpenAI形式に変換
7. **クライアントへ返却**: 変換したレスポンスをクライアントに返却

### パススルーモード

TCGWは以下の流れで動作します：

1. **リクエスト受信**: クライアントからOpenAI形式のリクエストを受信
2. **バックエンドへの転送**: リクエストをそのままBifrostに転送
3. **レスポンス受信**: BifrostからLLMの応答を受信
4. **クライアントへ返却**: レスポンスをそのままクライアントに返却

### ツール呼び出しの形式（エミュレートモード）

LLMは以下のようなXML形式でツール呼び出しを出力する必要があります：

```xml
<function_calls>
  <invoke name="get_weather">
    <parameter name="city">東京</parameter>
    <parameter name="units">celsius</parameter>
  </invoke>
</function_calls>
```

TCGWはこれを自動的に以下のJSON形式に変換します：

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\":\"東京\",\"units\":\"celsius\"}"
  }
}
```

## 高度な機能

### 複数ツールの同時呼び出し（エミュレートモード）

TCGWは、1つのレスポンス内で複数のツール呼び出しをサポートします：

```xml
<function_calls>
  <invoke name="get_weather">
    <parameter name="city">東京</parameter>
  </invoke>
  <invoke name="get_weather">
    <parameter name="city">大阪</parameter>
  </invoke>
</function_calls>
```

### パラメータの型推定（エミュレートモード）

TCGWは、パラメータの値を自動的に適切な型に変換します：

- **文字列**: `"Tokyo"` → `"Tokyo"`
- **整数**: `"10"` → `10`
- **浮動小数点数**: `"23.5"` → `23.5`
- **真偽値**: `"true"` → `true`

例：

```xml
<parameter name="city">Tokyo</parameter>         <!-- 文字列 -->
<parameter name="limit">10</parameter>           <!-- 整数 -->
<parameter name="threshold">0.85</parameter>     <!-- 浮動小数点数 -->
<parameter name="strict">true</parameter>        <!-- 真偽値 -->
```

これは以下のJSONに変換されます：

```json
{
  "city": "Tokyo",
  "limit": 10,
  "threshold": 0.85,
  "strict": true
}
```

### デバッグモード

`DEBUG_MODE=true`に設定すると、詳細なログが出力されます：

```bash
# .envファイル
DEBUG_MODE=true
```

出力例：

```
[TCGW] Request Received (Emulate Mode)
  Model: llama3
  Tool Count: 1
  Message Count: 2
  Has Stream: false

[TCGW] Embedding Tools
  System Prompt Len: 512
  Messages Count: 2

[TCGW] Tool Calls Extracted
  Count: 1

[TCGW] Response Generated (Emulate Mode)
  Finish Reason: tool_calls
  Tool Calls Count: 1
  Response ID: chatcmpl-abc123
```

## エラーハンドリング

TCGWは以下のHTTPステータスコードを返します：

| ステータスコード | 説明 |
|----------------|------|
| 200 | リクエスト成功 |
| 400 | 不正なJSONまたはリクエスト形式 |
| 501 | ストリーミングリクエスト（エミュレートモードでは未対応） |
| 500 | サーバー内部エラー |
| 502 | Bifrostからの不正なレスポンス |
| 503 | Bifrostへの接続失敗 |

エラーレスポンス例：

```json
{
  "error": {
    "message": "Invalid JSON: unexpected character",
    "type": "invalid_request_error",
    "code": "invalid_request"
  }
}
```

## 制限事項

- **ストリーミング未対応（エミュレートモードのみ）**: エミュレートモードでは、ストリーミングリクエスト（`stream: true`）には対応していません。ストリーミングリクエストを送信すると、501エラーが返されます。パススルーモードでは、バックエンドがサポートしていればストリーミングが利用可能です。
- **トークン使用量（エミュレートモードのみ）**: エミュレートモードでは、レスポンスの`usage`フィールドは常に0を返します。パススルーモードでは、バックエンドからの値がそのまま返されます。

## トラブルシューティング

### サーバーが起動しない

- `.env`ファイルの設定を確認してください
- `BIFROST_URL`が正しく設定されているか確認してください
- `EMULATE_PORT`と`PASSTHROUGH_PORT`が異なる値であることを確認してください
- ポート番号が他のプロセスで使用されていないか確認してください

### ツール呼び出しが抽出されない（エミュレートモード）

- `DEBUG_MODE=true`に設定して、LLMの出力を確認してください
- LLMが正しいXML形式でツール呼び出しを出力しているか確認してください
- ツール定義の`description`が明確であることを確認してください
- パススルーモード（ポート3001）を試して、問題がエミュレート処理にあるか確認してください

### Bifrostへの接続エラー

- Bifrostが起動しているか確認してください
- `BIFROST_URL`が正しいか確認してください（`http://`または`https://`で始まる必要があります）
- ファイアウォールやネットワーク設定を確認してください

### どちらのモードを使うべきか

- **Tool Calling非対応モデル**（Llama、Mistral、Gemmaなど）を使用する場合：エミュレートモード（ポート3000）を使用
- **Tool Calling対応モデル**（GPT-4、Claude、Geminiなど）を使用する場合：パススルーモード（ポート3001）を使用
- 不明な場合：まずパススルーモードを試し、動作しない場合はエミュレートモードを使用
