import { UnifiedChatRequest, UnifiedMessage, UnifiedTool } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import { log } from "@/utils/log";

export class OpenAIResponsesTransformer implements Transformer {
  name = "OpenAIResponses";
  endPoint = "/v1beta/responses";

  // Transform unified request to OpenAI Responses API format
  transformRequestOut(request: Record<string, any>): UnifiedChatRequest {
    log("OpenAI Responses Request:", JSON.stringify(request, null, 2));

    const messages: UnifiedMessage[] = [];
    
    // Handle instructions (system message)
    if (request.instructions) {
      messages.push({
        role: "system",
        content: request.instructions,
      });
    }

    // Convert input items to unified messages
    if (request.input) {
      if (typeof request.input === "string") {
        messages.push({
          role: "user",
          content: request.input,
        });
      } else if (Array.isArray(request.input)) {
        request.input.forEach((item: any) => {
          if (item.type === "message") {
            messages.push({
              role: item.role || "user",
              content: this.convertContentToUnified(item.content),
            });
          } else if (item.type === "function_call") {
            messages.push({
              role: "assistant",
              content: null,
              tool_calls: [{
                id: item.call_id,
                type: "function",
                function: {
                  name: item.name,
                  arguments: item.arguments,
                },
              }],
            });
          } else if (item.type === "function_call_output") {
            messages.push({
              role: "tool",
              content: item.output,
              tool_call_id: item.call_id,
            });
          }
        });
      }
    }

    const result: UnifiedChatRequest = {
      messages,
      model: request.model,
      max_tokens: request.max_output_tokens,
      temperature: request.temperature,
      stream: request.stream,
      tools: request.tools ? this.convertToolsToUnified(request.tools) : undefined,
      tool_choice: this.convertToolChoice(request.tool_choice),
    };

    return result;
  }

  // Transform response from OpenAI Responses API to unified format
  async transformResponseIn(response: Response): Promise<Response> {
    const isStream = response.headers
      .get("Content-Type")
      ?.includes("text/event-stream");

    if (isStream) {
      if (!response.body) {
        throw new Error("Stream response body is null");
      }
      const convertedStream = await this.convertResponsesStreamToUnified(
        response.body
      );
      return new Response(convertedStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      const data = await response.json();
      const unifiedResponse = this.convertResponsesToUnified(data);
      return new Response(JSON.stringify(unifiedResponse), {
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  private convertContentToUnified(content: any): string | null | any[] {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content.map((item) => {
        if (item.type === "input_text") {
          return { type: "text", text: item.text };
        } else if (item.type === "input_image") {
          return {
            type: "image",
            image_url: {
              url: item.image_url || item.file_id,
              detail: item.detail || "auto",
            },
          };
        }
        return item;
      });
    }
    return null;
  }

  private convertToolsToUnified(tools: any[]): UnifiedTool[] {
    return tools
      .filter(tool => tool.type === "function")
      .map((tool) => ({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description || "",
          parameters: tool.parameters || {},
        },
      }));
  }

  private convertToolChoice(toolChoice: any): any {
    if (!toolChoice) return undefined;
    if (typeof toolChoice === "string") {
      return toolChoice;
    }
    if (toolChoice.type === "function") {
      return toolChoice.name;
    }
    return "auto";
  }

  // Convert Responses API stream to unified stream format
  private async convertResponsesStreamToUnified(
    responsesStream: ReadableStream
  ): Promise<ReadableStream> {
    const readable = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        const decoder = new TextDecoder();
        let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
        
        // State tracking
        let currentMessage: any = null;
        let currentToolCalls: Map<string, any> = new Map();
        let accumulatedText = "";
        let accumulatedToolArgs: Map<string, string> = new Map();
        let hasStarted = false;
        let messageId = "";
        let model = "";
        let chunkId = 0;

        const createChunk = (delta: any, finishReason?: string) => {
          return {
            id: messageId || `chatcmpl-${Date.now()}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: model || "gpt-4",
            choices: [{
              index: 0,
              delta,
              finish_reason: finishReason || null,
            }],
          };
        };

        try {
          reader = responsesStream.getReader();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;

              // Parse SSE format
              if (line.startsWith("event: ")) {
                const eventType = line.slice(7).trim();
                continue; // Event type will be used with next data line
              }

              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6);
              
              try {
                const event = JSON.parse(data);
                log("Responses API Event:", event.type, event);

                switch (event.type) {
                  case "response.created":
                    messageId = event.response.id;
                    model = event.response.model;
                    if (!hasStarted) {
                      hasStarted = true;
                      // Send initial chunk
                      const chunk = createChunk({ role: "assistant", content: "" });
                      controller.enqueue(
                        encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
                      );
                    }
                    break;

                  case "response.output_item.added":
                    if (event.item.type === "message") {
                      // Message item added
                      currentMessage = event.item;
                    } else if (event.item.type === "function_call") {
                      // Function call added
                      const toolCall = {
                        id: event.item.call_id,
                        type: "function",
                        function: {
                          name: event.item.name,
                          arguments: "",
                        },
                      };
                      currentToolCalls.set(event.item.id, toolCall);
                    }
                    break;

                  case "response.content_part.added":
                    // Content part added (text or refusal)
                    break;

                  case "response.output_text.delta":
                    // Text delta
                    if (event.delta) {
                      const chunk = createChunk({ content: event.delta });
                      controller.enqueue(
                        encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
                      );
                    }
                    break;

                  case "response.function_call_arguments.delta":
                    // Function arguments delta
                    if (event.delta && event.item_id) {
                      const existingArgs = accumulatedToolArgs.get(event.item_id) || "";
                      accumulatedToolArgs.set(event.item_id, existingArgs + event.delta);
                      
                      const toolCall = currentToolCalls.get(event.item_id);
                      if (toolCall) {
                        const chunk = createChunk({
                          tool_calls: [{
                            index: 0,
                            id: toolCall.id,
                            type: "function",
                            function: {
                              name: toolCall.function.name,
                              arguments: event.delta,
                            },
                          }],
                        });
                        controller.enqueue(
                          encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
                        );
                      }
                    }
                    break;

                  case "response.output_text.done":
                    // Text completed
                    accumulatedText = event.text || "";
                    break;

                  case "response.function_call_arguments.done":
                    // Function arguments completed
                    if (event.item_id) {
                      const toolCall = currentToolCalls.get(event.item_id);
                      if (toolCall) {
                        toolCall.function.arguments = event.arguments;
                      }
                    }
                    break;

                  case "response.output_item.done":
                    // Output item completed
                    break;

                  case "response.completed":
                    // Response completed
                    const finishChunk = createChunk({}, "stop");
                    controller.enqueue(
                      encoder.encode(`data: ${JSON.stringify(finishChunk)}\n\n`)
                    );
                    controller.enqueue(encoder.encode("data: [DONE]\n\n"));
                    break;

                  case "response.failed":
                    // Handle error
                    const errorChunk = createChunk({}, "stop");
                    controller.enqueue(
                      encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`)
                    );
                    controller.enqueue(encoder.encode("data: [DONE]\n\n"));
                    break;

                  // Tool-specific events
                  case "response.file_search_call.in_progress":
                  case "response.file_search_call.searching":
                  case "response.file_search_call.completed":
                  case "response.web_search_call.in_progress":
                  case "response.web_search_call.searching":
                  case "response.web_search_call.completed":
                    // Handle tool progress events if needed
                    log(`Tool event: ${event.type}`);
                    break;

                  default:
                    log(`Unhandled event type: ${event.type}`);
                }

                chunkId++;
              } catch (parseError) {
                log("Parse error:", parseError);
              }
            }
          }
        } catch (error) {
          log("Stream processing error:", error);
          controller.error(error);
        } finally {
          if (reader) {
            try {
              reader.releaseLock();
            } catch (e) {
              log("Error releasing reader lock:", e);
            }
          }
          controller.close();
        }
      },
    });

    return readable;
  }

  // Convert non-streaming response
  private convertResponsesToUnified(responsesData: any): any {
    log("Original Responses API response:", JSON.stringify(responsesData, null, 2));

    const messages: any[] = [];
    let content = "";
    const toolCalls: any[] = [];

    // Process output items
    if (responsesData.output && Array.isArray(responsesData.output)) {
      responsesData.output.forEach((item: any) => {
        if (item.type === "message") {
          // Process message content
          if (item.content && Array.isArray(item.content)) {
            item.content.forEach((contentItem: any) => {
              if (contentItem.type === "output_text") {
                content += contentItem.text;
              } else if (contentItem.type === "refusal") {
                content += `[Refusal: ${contentItem.refusal}]`;
              }
            });
          }
        } else if (item.type === "function_call") {
          toolCalls.push({
            id: item.call_id || item.id,
            type: "function",
            function: {
              name: item.name,
              arguments: item.arguments,
            },
          });
        }
      });
    }

    // Use output_text if available
    if (responsesData.output_text) {
      content = responsesData.output_text;
    }

    const result = {
      id: responsesData.id,
      object: "chat.completion",
      created: responsesData.created_at || Math.floor(Date.now() / 1000),
      model: responsesData.model,
      choices: [{
        index: 0,
        message: {
          role: "assistant",
          content: content || null,
          ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
        },
        finish_reason: this.mapFinishReason(responsesData.status),
      }],
      usage: responsesData.usage ? {
        prompt_tokens: responsesData.usage.input_tokens || 0,
        completion_tokens: responsesData.usage.output_tokens || 0,
        total_tokens: responsesData.usage.total_tokens || 0,
      } : undefined,
    };

    log("Converted to unified format:", JSON.stringify(result, null, 2));
    return result;
  }

  private mapFinishReason(status: string): string {
    switch (status) {
      case "completed":
        return "stop";
      case "incomplete":
        return "length";
      case "failed":
        return "content_filter";
      default:
        return "stop";
    }
  }
}