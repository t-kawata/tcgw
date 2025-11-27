import OpenAI from "openai";

// LLMプロバイダーの設定を配列で定義
const llmProviders = [
  {
    name: "Bifrost OpenAI gpt-4o-mini (tool_call enabled)",
    apiKey: process.env.OPENAI_API_KEY_2,
    baseUrl: "http://0.0.0.0:7766/v1", // Bifrost
    model: "openai/gpt-4o-mini",
    enableToolCall: true, // ツール呼び出しを有効化
  },
  {
    name: "Bifrost OpenAI gpt-4o-mini (tool_call disabled)",
    apiKey: process.env.OPENAI_API_KEY_2,
    baseUrl: "http://0.0.0.0:7766/v1", // Bifrost
    model: "openai/gpt-4o-mini",
    enableToolCall: false, // ツール呼び出しを無効化
  },
  {
    name: "TCGW OpenAI gpt-4o-mini (tool_call enabled)",
    apiKey: process.env.OPENAI_API_KEY_2,
    baseUrl: "http://0.0.0.0:3001/v1", // TCGW
    model: "openai/gpt-4o-mini",
    enableToolCall: true, // ツール呼び出しを有効化
  },
  {
    name: "TCGW OpenAI gpt-4o-mini (tool_call disabled)",
    apiKey: process.env.OPENAI_API_KEY_2,
    baseUrl: "http://0.0.0.0:3001/v1", // TCGW
    model: "openai/gpt-4o-mini",
    enableToolCall: false, // ツール呼び出しを無効化
  },
  {
    name: "TCGW OpenAI gpt-4o-mini (tool_call enabled)",
    apiKey: process.env.OPENAI_API_KEY_2,
    baseUrl: "http://0.0.0.0:3000/v1", // TCGW
    model: "openai/gpt-4o-mini",
    enableToolCall: true, // ツール呼び出しを有効化
  },
  {
    name: "TCGW OpenAI gpt-4o-mini (tool_call disabled)",
    apiKey: process.env.OPENAI_API_KEY_2,
    baseUrl: "http://0.0.0.0:3000/v1", // TCGW
    model: "openai/gpt-4o-mini",
    enableToolCall: false, // ツール呼び出しを無効化
  },
];

// Sleep関数
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 3つのツール関数を定義
async function searchDatabase(query) {
  console.log(`\n    🟢 [Tool Execution] searchDatabase`);
  console.log(`           └─ Query: "${query}"`);
  // 単価情報を含む商品データを返す
  const result = {
    results: [
      { name: "商品A", unitPrice: 1000, description: "人気商品" },
      { name: "商品B", unitPrice: 1500, description: "高品質商品" },
      { name: "商品C", unitPrice: 800, description: "お買い得商品" },
    ],
  };
  console.log(`           └─ Result: ${JSON.stringify(result)}`);
  return result;
}

async function calculatePrice(quantity, unitPrice) {
  console.log(`\n    🟢 [Tool Execution] calculatePrice`);
  console.log(`           ├─ Quantity: ${quantity}`);
  console.log(`           ├─ Unit Price: ${unitPrice}`);
  const result = { total: quantity * unitPrice, currency: "JPY" };
  console.log(`           └─ Result: ${JSON.stringify(result)}`);
  return result;
}

async function sendNotification(message) {
  console.log(`\n    🟢 [Tool Execution] sendNotification`);
  console.log(`           └─ Message: "${message}"`);
  const result = { status: "sent", timestamp: new Date().toISOString() };
  console.log(`           └─ Result: ${JSON.stringify(result)}`);
  return result;
}

// OpenAI用のツール定義（3つ）
const tools = [
  {
    type: "function",
    function: {
      name: "searchDatabase",
      description: "データベースから商品情報を検索する。各商品には名前、単価、説明が含まれます。",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "検索クエリ（例：商品、在庫、価格など）",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculatePrice",
      description: "指定された数量と単価から合計金額を計算する",
      parameters: {
        type: "object",
        properties: {
          quantity: {
            type: "number",
            description: "購入数量",
          },
          unitPrice: {
            type: "number",
            description: "商品の単価（円）",
          },
        },
        required: ["quantity", "unitPrice"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "sendNotification",
      description: "ユーザーに通知メッセージを送信する",
      parameters: {
        type: "object",
        properties: {
          message: {
            type: "string",
            description: "送信する通知メッセージの内容",
          },
        },
        required: ["message"],
      },
    },
  },
];

// 利用可能なツールをマッピング
const availableTools = {
  searchDatabase,
  calculatePrice,
  sendNotification,
};

// エージェント関数（プロバイダー情報を受け取る）
async function agent(providerConfig, userInput) {
  const { name, apiKey, baseUrl, model, enableToolCall } = providerConfig;

  console.log(`\n${"=".repeat(80)}`);
  console.log(`[Provider] ${name}`);
  console.log(`   ├─ Base URL: ${baseUrl}`);
  console.log(`   ├─ Model: ${model}`);
  console.log(`   ├─ Tool Call: ${enableToolCall ? "ENABLED" : "DISABLED"}`);
  console.log(`   └─ User Input: "${userInput}"`);
  console.log(`${"=".repeat(80)}`);

  // プロバイダー固有のクライアントを初期化
  const openai = new OpenAI({
    apiKey: apiKey,
    baseURL: baseUrl,
  });

  const messages = [
    {
      role: "system",
      content:
        "あなたは役立つアシスタントです。提供されたツールのみを使用してください。" +
        "商品を検索したら、その結果に含まれる単価情報を使って価格計算を行ってください。",
    },
    {
      role: "user",
      content: userInput,
    },
  ];

  let iterationCount = 0;

  // 最大5回のループでツール呼び出しを処理
  for (let i = 0; i < 5; i++) {
    iterationCount++;
    console.log(`\n  [Iteration ${iterationCount}] Sending request to ${model}...`);

    // 2回目以降のイテレーションの前に1500ms待機
    if (i > 0) {
      console.log(`  [Sleep] Waiting 1500ms before next request...`);
      await sleep(1500);
    }

    try {
      // リクエストパラメータを構築（常にtoolsを含む）
      const requestParams = {
        model: model,
        messages: messages,
        tools: tools, // 常にツール定義を送信
      };

      // enableToolCallがfalseの場合、tool_choiceを"none"に設定
      if (!enableToolCall) {
        requestParams.tool_choice = "none";
        console.log(`  [Tool Choice] Set to "none" - tool calling disabled`);
      }

      const response = await openai.chat.completions.create(requestParams);

      const { finish_reason, message } = response.choices[0];
      console.log(`  [Response] Finish Reason: ${finish_reason}`);

      if (finish_reason === "tool_calls" && message.tool_calls) {
        // ツール呼び出しが要求された場合
        messages.push(message);

        console.log(`  [Tool Calls] ${message.tool_calls.length} tool(s) requested:`);

        for (const toolCall of message.tool_calls) {
          const functionName = toolCall.function.name;
          const functionToCall = availableTools[functionName];
          const functionArgs = JSON.parse(toolCall.function.arguments);
          const functionArgsArr = Object.values(functionArgs);

          console.log(`\n    ┌─ Tool Call ID: ${toolCall.id}`);
          console.log(`    ├─ Function: ${functionName}`);
          console.log(`    └─ Arguments: ${JSON.stringify(functionArgs)}`);

          // ツールを実行
          const functionResponse = await functionToCall.apply(null, functionArgsArr);

          // 結果をメッセージに追加
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify(functionResponse),
          });
        }
      } else if (finish_reason === "stop") {
        // 完了
        messages.push(message);
        console.log(`\n  ✅ [Final Response]`);
        console.log(`     ${message.content}`);
        console.log(`\n${"=".repeat(80)}\n`);
        return {
          provider: name,
          model: model,
          toolCallEnabled: enableToolCall,
          response: message.content,
          iterations: iterationCount,
        };
      }
    } catch (error) {
      console.error(`\n  ❌ [Error] ${error.message}`);
      console.log(`${"=".repeat(80)}\n`);
      return {
        provider: name,
        model: model,
        toolCallEnabled: enableToolCall,
        error: error.message,
        iterations: iterationCount,
      };
    }
  }

  console.log(`\n  ⚠️ [Warning] Maximum iterations reached`);
  console.log(`${"=".repeat(80)}\n`);
  return {
    provider: name,
    model: model,
    toolCallEnabled: enableToolCall,
    response: "最大反復回数に達しました。",
    iterations: iterationCount,
  };
}

// メイン処理：全プロバイダーを順番に実行
async function runAllProviders(userInput) {
  console.log(`\n╔${"═".repeat(78)}╗`);
  console.log(`║ 🤖 Multi-Provider LLM Tool Calling Test${" ".repeat(38)}║`);
  console.log(`╚${"═".repeat(78)}╝`);

  const results = [];

  for (let i = 0; i < llmProviders.length; i++) {
    const provider = llmProviders[i];

    // 2つ目以降のプロバイダーの前に1500ms待機
    if (i > 0) {
      console.log(`\n[Sleep] Waiting 1500ms before next provider...`);
      await sleep(1500);
    }

    const result = await agent(provider, userInput);
    results.push(result);
  }

  // 最終サマリーを表示
  console.log(`\n╔${"═".repeat(78)}╗`);
  console.log(`║ 📊 Summary of All Providers${" ".repeat(50)}║`);
  console.log(`╚${"═".repeat(78)}╝`);

  results.forEach((result, index) => {
    console.log(`\n[${index + 1}] ${result.provider}`);
    console.log(`    Model: ${result.model}`);
    console.log(`    Tool Call: ${result.toolCallEnabled ? "ENABLED" : "DISABLED"}`);
    console.log(`    Iterations: ${result.iterations}`);
    if (result.error) {
      console.log(`    Status: ❌ Error - ${result.error}`);
    } else {
      console.log(`    Status: ✅ Success`);
      console.log(`    Response: ${result.response.substring(0, 100)}...`);
    }
  });

  return results;
}

// 実行例
const userInput = "データベースで商品Cを検索して、その価格を3個分計算し、結果を通知してください";
const results = await runAllProviders(userInput);
