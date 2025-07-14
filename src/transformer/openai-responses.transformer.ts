import { log } from "../utils/log";
import {
  LLMProvider,
  UnifiedChatRequest,
  UnifiedMessage,
  UnifiedTool,
} from "@/types/llm";
import { Transformer } from "../types/transformer";

// transformRequestIn の冒頭か別関数として定義
function sanitizeFormatsRecursively(schema: any) {
  if (!schema || typeof schema !== "object") return;

  // ← ここで root も含め全レベルで format: "uri" を除去
  if (schema.format === "uri") {
    delete schema.format;
  }

  // properties があれば各プロパティをチェック
  if (schema.properties && typeof schema.properties === "object") {
    for (const key of Object.keys(schema.properties)) {
      const prop = schema.properties[key];
      // 'uri' フォーマットを削除
      if (prop.format === "uri") {
        delete prop.format;
      }
      // 再帰処理
      sanitizeFormatsRecursively(prop);
    }
  }

  // array の items に対しても再帰
  if (schema.type === "array" && schema.items) {
    sanitizeFormatsRecursively(schema.items);
  }
}

// transformRequestIn の冒頭か別関数として定義
function fillRequiredRecursively(schema: any) {
  if (!schema || typeof schema !== "object") return;

  // properties を持つ object スキーマには必ず required を補完
  if (schema.properties && typeof schema.properties === "object") {
    const keys = Object.keys(schema.properties);
    if (!Array.isArray(schema.required)) {
      schema.required = [...keys];
    } else {
      // 重複を避けつつマージ
      const missing = keys.filter((k) => !schema.required.includes(k));
      schema.required.push(...missing);
    }
    // ネストされた各 property を再帰
    keys.forEach((k) => fillRequiredRecursively(schema.properties[k]));
  }

  // array の items に対しても同様
  if (schema.type === "array" && schema.items) {
    fillRequiredRecursively(schema.items);
  }
}

export class OpenAIResponsesTransformer implements Transformer {
  name = "OpenAIResponses";

  endPoint = "/v1/responses";

  // Tool call IDマッピングを保持
  private toolCallIdMap = new Map<string, string>();

  transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider
  ): Record<string, any> {
    // 新しいリクエストではマッピングをクリア
    this.toolCallIdMap.clear();

    // Convert UnifiedChatRequest to OpenAI Responses API format
    const inputItems: any[] = [];

    // Process messages into input items
    request.messages.forEach((message) => {
      if (message.role === "system" || message.role === "user") {
        const content = this.convertMessageContent(message);
        inputItems.push({
          type: "message",
          role: message.role === "system" ? "developer" : message.role,
          content: content,
        });
      } else if (message.role === "assistant") {
        // Assistant messages with content
        if (message.content) {
          const content = this.convertMessageContent(message);
          inputItems.push({
            type: "message",
            role: "assistant",
            content: content,
          });
        }

        // Handle tool calls
        if (message.tool_calls && message.tool_calls.length > 0) {
          message.tool_calls.forEach((toolCall) => {
            // OpenAI Responses APIは fc_ プレフィックスを要求
            // Claudeのフォーマットに合わせて長いIDを生成
            const randomPart = Array.from(
              { length: 32 },
              () => Math.random().toString(36).charAt(2) || "0"
            ).join("");
            const newCallId = `fc_${randomPart}`;
            this.toolCallIdMap.set(toolCall.id, newCallId);

            inputItems.push({
              type: "function_call",
              id: newCallId,
              call_id: newCallId,
              name: toolCall.function.name,
              arguments: toolCall.function.arguments,
            });
          });
        }
      } else if (message.role === "tool") {
        // Tool results - マッピングされたIDを使用
        const mappedCallId =
          this.toolCallIdMap.get(message.tool_call_id!) ||
          message.tool_call_id!;
        inputItems.push({
          type: "function_call_output",
          call_id: mappedCallId,
          output:
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message.content),
        });
      }
    });

    // Build the request body
    const body: any = {
      input: inputItems,
      model: request.model,
      stream: request.stream || false,
    };

    // Add optional parameters
    // if (request.temperature !== undefined) {
    //   body.temperature = request.temperature;
    // }
    if (request.max_tokens !== undefined) {
      body.max_output_tokens = request.max_tokens;
    }

    // transformRequestIn 内の tools マッピング部
    if (request.tools && request.tools.length > 0) {
      body.tools = request.tools.map((tool) => {
        const params = tool.function.parameters;
        sanitizeFormatsRecursively(params);
        fillRequiredRecursively(params);
        return {
          type: "function",
          name: tool.function.name,
          description: tool.function.description,
          parameters: params,
          strict: true,
        };
      });
    }

    // Handle tool_choice
    if (request.tool_choice) {
      if (
        request.tool_choice === "auto" ||
        request.tool_choice === "none" ||
        request.tool_choice === "required"
      ) {
        body.tool_choice = request.tool_choice;
      } else {
        // Specific function
        body.tool_choice = {
          type: "function",
          name: request.tool_choice,
        };
      }
    }

    return {
      body,
      config: {
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
        },
      },
    };
  }

  private convertMessageContent(message: UnifiedMessage): string | any[] {
    if (typeof message.content === "string") {
      return message.content;
    } else if (Array.isArray(message.content)) {
      return message.content.map((item) => {
        if (item.type === "text") {
          return {
            type: "input_text",
            text: item.text,
          };
        } else if (item.type === "image") {
          return {
            type: "input_image",
            image_url: item.image_url,
          };
        }
        return item;
      });
    }
    return "";
  }

  transformRequestOut(request: Record<string, any>): UnifiedChatRequest {
    const messages: UnifiedMessage[] = [];

    if (request.input && Array.isArray(request.input)) {
      request.input.forEach((item: any) => {
        if (item.type === "message") {
          const content = this.convertResponseContent(item.content);
          messages.push({
            role: item.role === "developer" ? "system" : item.role,
            content: content,
          });
        } else if (item.type === "function_call") {
          // Find or create the last assistant message
          let lastAssistant = messages[messages.length - 1];
          if (!lastAssistant || lastAssistant.role !== "assistant") {
            lastAssistant = {
              role: "assistant",
              content: null,
              tool_calls: [],
            };
            messages.push(lastAssistant);
          }

          if (!lastAssistant.tool_calls) {
            lastAssistant.tool_calls = [];
          }

          // 元のIDに戻すためのリバースマッピング
          const originalId =
            Array.from(this.toolCallIdMap.entries()).find(
              ([, mappedId]) => mappedId === (item.id || item.call_id)
            )?.[0] ||
            item.id ||
            item.call_id;

          lastAssistant.tool_calls.push({
            id: originalId,
            type: "function",
            function: {
              name: item.name,
              arguments: item.arguments,
            },
          });
        } else if (item.type === "function_call_output") {
          // 元のIDに戻すためのリバースマッピング
          const originalId =
            Array.from(this.toolCallIdMap.entries()).find(
              ([, mappedId]) => mappedId === item.call_id
            )?.[0] || item.call_id;

          messages.push({
            role: "tool",
            content: item.output,
            tool_call_id: originalId,
          });
        }
      });
    }

    const result: UnifiedChatRequest = {
      messages,
      model: request.model,
      stream: request.stream,
    };

    // if (request.temperature !== undefined) {
    //   result.temperature = request.temperature;
    // }
    if (request.max_output_tokens !== undefined) {
      result.max_tokens = request.max_output_tokens;
    }

    // Convert tools
    if (request.tools && Array.isArray(request.tools)) {
      result.tools = request.tools.map((tool: any) => ({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description || "",
          parameters: tool.parameters,
        },
      }));
    }

    // Convert tool_choice
    if (request.tool_choice) {
      if (typeof request.tool_choice === "string") {
        result.tool_choice = request.tool_choice;
      } else if (request.tool_choice.type === "function") {
        result.tool_choice = request.tool_choice.name;
      }
    }

    return result;
  }

  private convertResponseContent(content: any): string | any[] {
    if (typeof content === "string") {
      return content;
    } else if (Array.isArray(content)) {
      const textParts: string[] = [];
      const otherParts: any[] = [];

      content.forEach((item) => {
        if (item.type === "input_text" || item.type === "output_text") {
          textParts.push(item.text);
        } else if (item.type === "input_image") {
          otherParts.push({
            type: "image",
            image_url: item.image_url,
          });
        }
      });

      if (otherParts.length === 0) {
        return textParts.join("");
      } else {
        const result: any[] = [];
        if (textParts.length > 0) {
          result.push({ type: "text", text: textParts.join("") });
        }
        result.push(...otherParts);
        return result;
      }
    }
    return "";
  }

  async transformResponseOut(response: Response): Promise<Response> {
    if (response.headers.get("Content-Type")?.includes("application/json")) {
      // Handle non-streaming response
      const jsonResponse = (await response.json()) as any;

      // Transform to Chat Completions format
      const choices: any[] = [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
          },
          finish_reason: null,
        },
      ];

      let hasContent = false;
      const toolCalls: any[] = [];

      if (jsonResponse.output && Array.isArray(jsonResponse.output)) {
        jsonResponse.output.forEach((item: any) => {
          if (item.type === "message" && item.role === "assistant") {
            const content = this.extractTextFromContent(item.content);
            if (content) {
              choices[0].message.content = content;
              hasContent = true;
            }
          } else if (item.type === "function_call") {
            toolCalls.push({
              id:
                Array.from(this.toolCallIdMap.entries()).find(
                  ([, mappedId]) => mappedId === (item.id || item.call_id)
                )?.[0] ||
                item.id ||
                item.call_id,
              type: "function",
              function: {
                name: item.name,
                arguments: item.arguments,
              },
            });
          }
        });
      }

      if (toolCalls.length > 0) {
        choices[0].message.tool_calls = toolCalls;
        if (!hasContent) {
          choices[0].message.content = null;
        }
      }

      // Set finish reason
      if (jsonResponse.status === "completed") {
        choices[0].finish_reason = toolCalls.length > 0 ? "tool_calls" : "stop";
      } else if (jsonResponse.status === "failed") {
        choices[0].finish_reason = "stop";
      }

      const transformedResponse = {
        id: jsonResponse.id,
        object: "chat.completion",
        created: jsonResponse.created_at || Math.floor(Date.now() / 1000),
        model: jsonResponse.model,
        choices: choices,
        usage: jsonResponse.usage
          ? {
              prompt_tokens: jsonResponse.usage.input_tokens,
              completion_tokens: jsonResponse.usage.output_tokens,
              total_tokens: jsonResponse.usage.total_tokens,
            }
          : undefined,
      };

      return new Response(JSON.stringify(transformedResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } else if (response.headers.get("Content-Type")?.includes("stream")) {
      if (!response.body) {
        return response;
      }

      const decoder = new TextDecoder();
      const encoder = new TextEncoder();

      // State tracking
      let currentContent = "";
      let currentToolCalls: Map<string, any> = new Map();
      let responseId = "";
      let model = "";
      let isFirstChunk = true;
      let choiceIndex = 0; // ← 追加: choice indexの管理

      // IDマッピングをコピー
      const toolCallIdMap = new Map(this.toolCallIdMap);

      const stream = new ReadableStream({
        async start(controller) {
          const reader = response.body!.getReader();
          let buffer = ""; // バッファを追加して不完全なデータを蓄積

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              buffer += chunk; // バッファに追加
              const lines = buffer.split("\n");

              // 最後の行は不完全な可能性があるので、バッファに残す
              buffer = lines.pop() || "";

              for (const line of lines) {
                if (
                  line.startsWith("data: ") &&
                  line.trim() !== "data: [DONE]"
                ) {
                  const jsonStr = line.slice(6).trim();
                  if (jsonStr === "") {
                    // 空のdata行をスキップ
                    continue;
                  }

                  try {
                    const eventData = JSON.parse(jsonStr);

                    switch (eventData.type) {
                      case "response.created":
                        responseId = eventData.response.id;
                        model = eventData.response.model;

                        const initialChunk = {
                          id: responseId,
                          object: "chat.completion.chunk",
                          created: eventData.response.created_at,
                          model: model,
                          choices: [
                            {
                              index: 0,
                              delta: { role: "assistant", content: "" },
                              finish_reason: null,
                            },
                          ],
                        };
                        controller.enqueue(
                          encoder.encode(
                            `data: ${JSON.stringify(initialChunk)}\n\n`
                          )
                        );
                        break;

                      case "response.output_text.delta":
                      case "response.text.delta":
                        if (eventData.delta) {
                          const deltaChunk = {
                            id: responseId,
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: model,
                            choices: [
                              {
                                index: choiceIndex,
                                delta: { content: eventData.delta },
                                finish_reason: null,
                              },
                            ],
                          };
                          controller.enqueue(
                            encoder.encode(
                              `data: ${JSON.stringify(deltaChunk)}\n\n`
                            )
                          );
                        }
                        break;

                      case "response.function_call_arguments.delta":
                        if (eventData.item_id && eventData.delta) {
                          // 元のIDにリバースマッピング
                          const originalId =
                            Array.from(toolCallIdMap.entries()).find(
                              ([, mappedId]) => mappedId === eventData.item_id
                            )?.[0] || eventData.item_id;

                          let toolCall = currentToolCalls.get(originalId);
                          if (!toolCall) {
                            // Tool callが開始されたら次のchoice indexに移る
                            if (currentContent) {
                              choiceIndex++;
                            }

                            toolCall = {
                              id: originalId,
                              type: "function",
                              function: {
                                name: "",
                                arguments: "",
                              },
                            };
                            currentToolCalls.set(originalId, toolCall);

                            const toolCallIndex = currentToolCalls.size - 1;
                            const toolCallChunk = {
                              id: responseId,
                              object: "chat.completion.chunk",
                              created: Math.floor(Date.now() / 1000),
                              model: model,
                              choices: [
                                {
                                  index: choiceIndex,
                                  delta: {
                                    tool_calls: [
                                      {
                                        index: toolCallIndex,
                                        id: originalId,
                                        type: "function",
                                        function: { name: "", arguments: "" },
                                      },
                                    ],
                                  },
                                  finish_reason: null,
                                },
                              ],
                            };
                            controller.enqueue(
                              encoder.encode(
                                `data: ${JSON.stringify(toolCallChunk)}\n\n`
                              )
                            );
                          }

                          toolCall.function.arguments += eventData.delta;

                          const toolCallIndex = Array.from(
                            currentToolCalls.keys()
                          ).indexOf(originalId);
                          const argsDeltaChunk = {
                            id: responseId,
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: model,
                            choices: [
                              {
                                index: choiceIndex,
                                delta: {
                                  tool_calls: [
                                    {
                                      index: toolCallIndex,
                                      function: {
                                        arguments: eventData.delta,
                                      },
                                    },
                                  ],
                                },
                                finish_reason: null,
                              },
                            ],
                          };
                          controller.enqueue(
                            encoder.encode(
                              `data: ${JSON.stringify(argsDeltaChunk)}\n\n`
                            )
                          );
                        }
                        break;

                      case "response.function_call_arguments.done":
                        if (eventData.item_id && eventData.name) {
                          // 元のIDにリバースマッピング
                          const originalId =
                            Array.from(toolCallIdMap.entries()).find(
                              ([, mappedId]) => mappedId === eventData.item_id
                            )?.[0] || eventData.item_id;

                          const toolCall = currentToolCalls.get(originalId);
                          if (toolCall) {
                            toolCall.function.name = eventData.name;

                            // Function name deltaを送信
                            const toolCallIndex = Array.from(
                              currentToolCalls.keys()
                            ).indexOf(originalId);
                            const nameDeltaChunk = {
                              id: responseId,
                              object: "chat.completion.chunk",
                              created: Math.floor(Date.now() / 1000),
                              model: model,
                              choices: [
                                {
                                  index: choiceIndex,
                                  delta: {
                                    tool_calls: [
                                      {
                                        index: toolCallIndex,
                                        function: {
                                          name: eventData.name,
                                        },
                                      },
                                    ],
                                  },
                                  finish_reason: null,
                                },
                              ],
                            };
                            controller.enqueue(
                              encoder.encode(
                                `data: ${JSON.stringify(nameDeltaChunk)}\n\n`
                              )
                            );
                          }
                        }
                        break;

                      case "response.completed":
                        const finalChunk = {
                          id: responseId,
                          object: "chat.completion.chunk",
                          created: Math.floor(Date.now() / 1000),
                          model: model,
                          choices: [
                            {
                              index: choiceIndex,
                              delta: {},
                              finish_reason:
                                currentToolCalls.size > 0
                                  ? "tool_calls"
                                  : "stop",
                            },
                          ],
                        };
                        controller.enqueue(
                          encoder.encode(
                            `data: ${JSON.stringify(finalChunk)}\n\n`
                          )
                        );
                        // [DONE]は最後に送信
                        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
                        return; // ← ここで処理終了
                    }
                  } catch (e) {
                    log("Error parsing Responses API event:", e);
                    log("Problematic line:", line);
                    log("JSON string:", jsonStr);
                    // JSONパースエラーの場合、この行をスキップして続行
                    continue;
                  }
                }
              }
            }

            // 処理完了時に残りのバッファをチェック
            if (buffer.trim() !== "") {
              log("Remaining buffer at end:", buffer);
            }
          } catch (error) {
            controller.error(error);
          } finally {
            try {
              reader.releaseLock();
            } catch (e) {
              console.error("Error releasing reader lock:", e);
            }
            controller.close();
          }
        },
      });

      return new Response(stream, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    return response;
  }

  private extractTextFromContent(content: any): string | null {
    if (typeof content === "string") {
      return content;
    } else if (Array.isArray(content)) {
      const textParts = content
        .filter(
          (item) => item.type === "output_text" || item.type === "input_text"
        )
        .map((item) => item.text);
      return textParts.length > 0 ? textParts.join("") : null;
    }
    return null;
  }
}
