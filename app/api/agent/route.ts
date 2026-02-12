import { createOpenAI } from "@ai-sdk/openai";
import { playwrightExecuteTool } from "@onkernel/ai-sdk";
import { Kernel } from "@onkernel/sdk";
import { Experimental_Agent as Agent, stepCountIs, tool } from "ai";
import { chromium } from "playwright-core";
import { z } from "zod";
import {
  buildMemoryPromptContext,
  loadRecentSessionMemory,
  saveSessionMemory,
} from "@/lib/supabase-memory";
import {
  getVpsSandboxConfig,
  resolveSandboxProvider,
} from "@/lib/sandbox-provider";

export const maxDuration = 300; // 5 minutes timeout for long-running agent operations

interface OllamaRuntimeConfig {
  provider: "local" | "cloud";
  baseURL: string;
  apiKey: string;
  models: {
    automation: string;
    reasoning: string;
    ocr: string;
  };
  headers?: Record<string, string>;
}

interface ModelSelection {
  role: "automation" | "reasoning" | "ocr";
  model: string;
}

const DEFAULT_MODELS = {
  automation: "kimi-k2-thinking:cloud",
  reasoning: "deepseek-v3.1:671b-cloud",
  ocr: "glm-ocr:latest",
} as const;

function buildModelsEndpoint(baseURL: string) {
  return `${baseURL.replace(/\/+$/, "")}/models`;
}

async function isOpenAICompatibleEndpointReachable(
  baseURL: string,
  apiKey: string,
  headers?: Record<string, string>
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2200);

  try {
    const response = await fetch(buildModelsEndpoint(baseURL), {
      method: "GET",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        ...(headers || {}),
      },
      signal: controller.signal,
      cache: "no-store",
    });

    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function resolveOllamaRuntimeConfig(): Promise<OllamaRuntimeConfig> {
  const localBaseURL = process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1";
  const localApiKey = process.env.OLLAMA_API_KEY || "ollama";
  const localModels = {
    automation:
      process.env.OLLAMA_MODEL_AUTOMATION ||
      process.env.OLLAMA_MODEL ||
      DEFAULT_MODELS.automation,
    reasoning: process.env.OLLAMA_MODEL_REASONING || DEFAULT_MODELS.reasoning,
    ocr: process.env.OLLAMA_MODEL_OCR || DEFAULT_MODELS.ocr,
  };

  const cloudBaseURL = process.env.OLLAMA_CLOUD_BASE_URL;
  const cloudApiKey = process.env.OLLAMA_CLOUD_API_KEY || localApiKey;
  const cloudDefaultModel = process.env.OLLAMA_CLOUD_MODEL;
  const cloudModels = {
    automation:
      process.env.OLLAMA_CLOUD_MODEL_AUTOMATION ||
      cloudDefaultModel ||
      localModels.automation,
    reasoning:
      process.env.OLLAMA_CLOUD_MODEL_REASONING ||
      DEFAULT_MODELS.reasoning,
    ocr:
      process.env.OLLAMA_CLOUD_MODEL_OCR ||
      DEFAULT_MODELS.ocr,
  };
  const cloudDeploymentSha = process.env.OLLAMA_CLOUD_DEPLOYMENT_SHA;
  const cloudSshPublicKey = process.env.OLLAMA_CLOUD_SSH_PUBLIC_KEY;

  const localReachable = await isOpenAICompatibleEndpointReachable(
    localBaseURL,
    localApiKey
  );
  if (localReachable) {
    return {
      provider: "local",
      baseURL: localBaseURL,
      apiKey: localApiKey,
      models: localModels,
    };
  }

  if (cloudBaseURL && cloudApiKey) {
    const cloudHeaders: Record<string, string> = {};
    if (cloudDeploymentSha) {
      cloudHeaders["x-deployment-sha"] = cloudDeploymentSha;
    }
    if (cloudSshPublicKey) {
      cloudHeaders["x-ssh-public-key"] = cloudSshPublicKey;
    }

    return {
      provider: "cloud",
      baseURL: cloudBaseURL,
      apiKey: cloudApiKey,
      models: cloudModels,
      headers: Object.keys(cloudHeaders).length ? cloudHeaders : undefined,
    };
  }

  // Keep local as last-resort behavior if cloud fallback is not configured.
  return {
    provider: "local",
    baseURL: localBaseURL,
    apiKey: localApiKey,
    models: localModels,
  };
}

function selectModelForTask(
  task: string,
  models: OllamaRuntimeConfig["models"]
): ModelSelection {
  const normalizedTask = task.toLowerCase();

  const isOCRTask =
    /\b(ocr|image|screenshot|scan|extract text|read text|vision|pdf)\b/i.test(
      normalizedTask
    );
  if (isOCRTask) {
    return { role: "ocr", model: models.ocr };
  }

  const isReasoningTask =
    /\b(analy[sz]e|analysis|summari[sz]e|compare|explain|research|plan|reason)\b/i.test(
      normalizedTask
    );
  if (isReasoningTask) {
    return { role: "reasoning", model: models.reasoning };
  }

  return { role: "automation", model: models.automation };
}

async function normalizeToSinglePageKernel(kernel: Kernel, sessionId: string) {
  try {
    await kernel.browsers.playwright.execute(sessionId, {
      timeout_sec: 20,
      code: `
const pages = context.pages();
if (!pages.length) {
  return { pageCountBefore: 0, pageCountAfter: 0 };
}

const primaryPage =
  pages.find(
    (p) =>
      !p.url().startsWith("chrome-extension://") &&
      p.url() !== "about:blank"
  ) ?? pages[0];

for (const currentPage of pages) {
  if (currentPage !== primaryPage) {
    try {
      await currentPage.close();
    } catch {}
  }
}

try {
  await primaryPage.bringToFront();
} catch {}

return {
  pageCountBefore: pages.length,
  pageCountAfter: context.pages().length,
  primaryUrl: primaryPage.url(),
};
      `,
    });
  } catch (error) {
    console.warn("Failed to normalize browser pages before agent run:", error);
  }
}

const computerUseInputSchema = z.object({
  action: z.enum([
    "navigate",
    "click",
    "move",
    "type_text",
    "key_press",
    "scroll",
    "wait",
    "screenshot",
  ]),
  url: z.string().url().optional(),
  x: z.number().optional(),
  y: z.number().optional(),
  button: z.enum(["left", "middle", "right"]).optional(),
  text: z.string().optional(),
  key: z.string().optional(),
  deltaX: z.number().optional(),
  deltaY: z.number().optional(),
  waitMs: z.number().int().min(0).max(30000).optional(),
  fullPage: z.boolean().optional(),
});

function getAsyncFunctionConstructor() {
  return Object.getPrototypeOf(async function () {
    // noop
  }).constructor as new (
    ...args: string[]
  ) => (...params: unknown[]) => Promise<unknown>;
}

function selectPrimaryPage(pages: Array<{ url(): string }>) {
  return (
    pages.find(
      (page) =>
        !page.url().startsWith("chrome-extension://") && page.url() !== "about:blank"
    ) ?? pages[0]
  );
}

async function withVpsBrowser<T>(
  cdpWsUrl: string,
  run: (context: any, page: any, browser: any) => Promise<T>
) {
  const browser = await chromium.connectOverCDP(cdpWsUrl);
  try {
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error("No browser context found on VPS CDP endpoint");
    }

    let page = selectPrimaryPage(context.pages());
    if (!page) {
      page = await context.newPage();
    }

    return await run(context, page, browser);
  } finally {
    await browser.close().catch(() => undefined);
  }
}

async function normalizeToSinglePageVps(cdpWsUrl: string) {
  try {
    await withVpsBrowser(cdpWsUrl, async (context, page) => {
      for (const currentPage of context.pages()) {
        if (currentPage !== page) {
          await currentPage.close().catch(() => undefined);
        }
      }
      await page.bringToFront().catch(() => undefined);
      return {
        pageCountAfter: context.pages().length,
      };
    });
  } catch (error) {
    console.warn("Failed to normalize VPS browser pages:", error);
  }
}

function createKernelComputerUseTool(kernel: Kernel, sessionId: string) {
  return tool({
    description:
      "Perform computer-level actions in the browser sandbox (mouse, keyboard, scroll, screenshot, wait, navigate).",
    inputSchema: computerUseInputSchema,
    execute: async (input) => {
      switch (input.action) {
        case "navigate": {
          if (!input.url) throw new Error("navigate action requires 'url'");
          return await kernel.browsers.playwright.execute(sessionId, {
            timeout_sec: 60,
            code: `await page.goto(${JSON.stringify(
              input.url
            )}, { waitUntil: "domcontentloaded" }); return { url: page.url(), title: await page.title() };`,
          });
        }
        case "click": {
          if (input.x === undefined || input.y === undefined) {
            throw new Error("click action requires both 'x' and 'y'");
          }
          await kernel.browsers.computer.clickMouse(sessionId, {
            x: input.x,
            y: input.y,
            button: input.button ?? "left",
            click_type: "click",
          });
          return { success: true };
        }
        case "move": {
          if (input.x === undefined || input.y === undefined) {
            throw new Error("move action requires both 'x' and 'y'");
          }
          await kernel.browsers.computer.moveMouse(sessionId, {
            x: input.x,
            y: input.y,
          });
          return { success: true };
        }
        case "type_text": {
          if (!input.text) throw new Error("type_text action requires 'text'");
          await kernel.browsers.computer.typeText(sessionId, {
            text: input.text,
          });
          return { success: true };
        }
        case "key_press": {
          if (!input.key) throw new Error("key_press action requires 'key'");
          await kernel.browsers.computer.pressKey(sessionId, {
            keys: [input.key],
          });
          return { success: true };
        }
        case "scroll": {
          await kernel.browsers.computer.scroll(sessionId, {
            x: input.x ?? 0,
            y: input.y ?? 0,
            delta_x: input.deltaX ?? 0,
            delta_y: input.deltaY ?? 400,
          });
          return { success: true };
        }
        case "wait": {
          const waitMs = input.waitMs ?? 1000;
          return await kernel.browsers.playwright.execute(sessionId, {
            timeout_sec: Math.ceil(waitMs / 1000) + 5,
            code: `await page.waitForTimeout(${waitMs}); return { waitedMs: ${waitMs} };`,
          });
        }
        case "screenshot": {
          const screenshotResponse = await kernel.browsers.computer.captureScreenshot(
            sessionId
          );
          const screenshotBuffer = await screenshotResponse.arrayBuffer();
          return {
            success: true,
            bytes: screenshotBuffer.byteLength,
            contentType:
              screenshotResponse.headers.get("content-type") || "image/png",
          };
        }
      }
    },
  });
}

function createVpsPlaywrightExecuteTool(cdpWsUrl: string) {
  return tool({
    description:
      "Execute JavaScript/Playwright code against the VPS browser session. Exposes page, context, and browser.",
    inputSchema: z.object({
      code: z.string().min(1),
    }),
    execute: async ({ code }) => {
      try {
        return await withVpsBrowser(cdpWsUrl, async (context, page, browser) => {
          const AsyncFunction = getAsyncFunctionConstructor();
          const executeCode = new AsyncFunction("page", "context", "browser", code);
          const result = await executeCode(page, context, browser);
          return { success: true, result };
        });
      } catch (error: any) {
        return {
          success: false,
          error: error?.message || "Failed to execute Playwright code on VPS",
        };
      }
    },
  });
}

function createVpsComputerUseTool(cdpWsUrl: string) {
  return tool({
    description:
      "Perform computer-level actions in the VPS browser (mouse, keyboard, scroll, screenshot, wait, navigate).",
    inputSchema: computerUseInputSchema,
    execute: async (input) => {
      return await withVpsBrowser(cdpWsUrl, async (_context, page) => {
        switch (input.action) {
          case "navigate": {
            if (!input.url) throw new Error("navigate action requires 'url'");
            await page.goto(input.url, { waitUntil: "domcontentloaded" });
            break;
          }
          case "click": {
            if (input.x === undefined || input.y === undefined) {
              throw new Error("click action requires both 'x' and 'y'");
            }
            await page.mouse.click(input.x, input.y, {
              button: input.button ?? "left",
            });
            break;
          }
          case "move": {
            if (input.x === undefined || input.y === undefined) {
              throw new Error("move action requires both 'x' and 'y'");
            }
            await page.mouse.move(input.x, input.y);
            break;
          }
          case "type_text": {
            if (!input.text) throw new Error("type_text action requires 'text'");
            await page.keyboard.type(input.text);
            break;
          }
          case "key_press": {
            if (!input.key) throw new Error("key_press action requires 'key'");
            await page.keyboard.press(input.key);
            break;
          }
          case "scroll": {
            await page.mouse.wheel(input.deltaX ?? 0, input.deltaY ?? 400);
            break;
          }
          case "wait": {
            await page.waitForTimeout(input.waitMs ?? 1000);
            break;
          }
          case "screenshot": {
            const screenshot = await page.screenshot({
              fullPage: input.fullPage ?? false,
              type: "jpeg",
              quality: 70,
            });
            return {
              success: true,
              bytes: screenshot.byteLength,
              screenshotPreviewBase64: screenshot.toString("base64").slice(0, 120),
              url: page.url(),
              title: await page.title(),
            };
          }
        }

        return {
          success: true,
          url: page.url(),
          title: await page.title(),
        };
      });
    },
  });
}

export async function POST(req: Request) {
  try {
    const { sessionId, task, cdpWsUrl, provider } = await req.json();
    const sandboxProvider = resolveSandboxProvider(provider);
    const vpsConfig = getVpsSandboxConfig();
    const memorySessionId =
      sessionId || vpsConfig?.sessionId || "vps-sandbox-session";

    if (!task) {
      return Response.json(
        { error: "Missing task" },
        { status: 400 }
      );
    }
    const ollamaRuntime = await resolveOllamaRuntimeConfig();
    const selectedModel = selectModelForTask(task, ollamaRuntime.models);
    const sessionMemory = await loadRecentSessionMemory(memorySessionId, 6);
    const memoryContext = buildMemoryPromptContext(sessionMemory);
    const ollama = createOpenAI({
      baseURL: ollamaRuntime.baseURL,
      apiKey: ollamaRuntime.apiKey,
      headers: ollamaRuntime.headers,
      name: "ollama",
    });

    let tools: any;
    if (sandboxProvider === "vps") {
      const resolvedCdpWsUrl = cdpWsUrl || vpsConfig?.cdpWsUrl;
      if (!resolvedCdpWsUrl) {
        return Response.json(
          {
            error:
              "VPS mode requires cdpWsUrl in request or VPS_SANDBOX_CDP_WS_URL in environment",
          },
          { status: 400 }
        );
      }

      await normalizeToSinglePageVps(resolvedCdpWsUrl);
      tools = {
        playwright_execute: createVpsPlaywrightExecuteTool(resolvedCdpWsUrl),
        computer_use: createVpsComputerUseTool(resolvedCdpWsUrl),
      };
    } else {
      const apiKey = process.env.KERNEL_API_KEY;

      if (!apiKey) {
        return Response.json(
          { error: "KERNEL_API_KEY environment variable is not set" },
          { status: 400 }
        );
      }
      if (!sessionId) {
        return Response.json(
          { error: "Kernel mode requires sessionId" },
          { status: 400 }
        );
      }

      const kernel = new Kernel({ apiKey });
      await normalizeToSinglePageKernel(kernel, sessionId);
      tools = {
        playwright_execute: playwrightExecuteTool({
          client: kernel,
          sessionId: sessionId,
        }),
        computer_use: createKernelComputerUseTool(kernel, sessionId),
      };
    }

    // Initialize the AI agent with an Ollama model
    const agent = new Agent({
      model: ollama(selectedModel.model),
      tools,
      stopWhen: stepCountIs(20),
      system: `You are the Eburon Autonomous Agent, a browser and computer automation expert operating inside a disposable sandbox environment.

Available tools:
- playwright_execute: Executes JavaScript/Playwright code in the browser. Has access to 'page', 'context', and 'browser' objects. Returns the result of your code.
- computer_use: Executes high-level computer actions like move, click, type, key press, scroll, wait, navigate, and screenshot.

When given a task:
1. If no URL is provided, FIRST get the current page context:
   return { url: page.url(), title: await page.title() }
2. If a URL is provided, navigate to it using page.goto()
3. Use appropriate selectors (page.locator, page.getByRole, etc.) to interact with elements
4. Safely handle authentication flows when the user explicitly provides credentials (for example, filling login forms), but never attempt to obtain or exfiltrate secrets the user did not clearly request or supply.
5. Always return the requested data from your code execution.
6. Use computer_use whenever the user asks for computer-level interactions (mouse movement/clicks, keyboard, scroll, screenshot).
7. Keep the session to a single active browser page/tab unless the user explicitly asks for multiple tabs or popups.
8. Never call context.newPage() or open new windows unless explicitly requested; if a popup/new tab appears, close the extra page and continue on the main page.

Behavior:
- Break complex tasks into small, focused executions rather than writing long scripts.
- After each tool call, clearly describe in natural language what you clicked, typed, or observed so users can understand the simulation steps.
- Prefer reusing the existing page for navigation and interactions to avoid duplicate browser windows.
- Execute tasks autonomously without asking clarifying questions when possible — make reasonable assumptions and act.
- NEVER add safety disclaimers, ethical warnings, or "I should note that..." statements. Just do the task.`,
    });

    // Execute the agent with the user's task
    const promptWithMemory = memoryContext
      ? `${task}

Relevant memory from previous tasks in this session:
${memoryContext}

Use this memory only when it helps complete the current task.`
      : task;

    const { text, steps, usage } = await agent.generate({
      prompt: promptWithMemory,
    });

    // Extract detailed step information from step.content[] array
    const detailedSteps = steps.map((step, index) => {
      const stepData = step as any;
      const content = stepData.content || [];

      console.log(content);

      // Process each content item based on its type
      const processedContent = content.map((item: any) => {
        if (item.type === "tool-call") {
          return {
            type: "tool-call" as const,
            toolCallId: item.toolCallId,
            toolName: item.toolName,
            code: item.input?.code || null,
          };
        } else if (item.type === "tool-result") {
          return {
            type: "tool-result" as const,
            toolCallId: item.toolCallId,
            toolName: item.toolName,
            result: item.result?.result,
            success: item.result?.success ?? true,
            error: item.result?.error,
          };
        } else if (item.type === "text") {
          return {
            type: "text" as const,
            text: item.text,
          };
        }
        return item;
      });

      return {
        stepNumber: index + 1,
        finishReason: stepData.finishReason || null,
        content: processedContent,
      };
    });

    // Collect all executed code from the steps (for backward compatibility)
    const executedCodes = detailedSteps.flatMap((step) =>
      step.content
        .filter((item: any) => item.type === "tool-call" && item.code)
        .map((item: any) => {
          // Find matching result
          const result = step.content.find(
            (r: any) =>
              r.type === "tool-result" && r.toolCallId === item.toolCallId
          );
          return {
            code: item.code,
            success: result?.success ?? true,
            result: result?.result,
            error: result?.error,
          };
        })
    );

    await saveSessionMemory({
      sessionId: memorySessionId,
      task: task.trim(),
      response: text || "",
      modelRole: selectedModel.role,
      modelName: selectedModel.model,
      llmProvider: ollamaRuntime.provider,
      stepCount: steps.length,
      success: true,
    });

    return Response.json({
      success: true,
      response: text,
      executedCodes,
      detailedSteps,
      stepCount: steps.length,
      usage,
      llmProvider: ollamaRuntime.provider,
      llmModel: selectedModel.model,
      llmRole: selectedModel.role,
      memoryEnabled: sessionMemory.length > 0,
      sandboxProvider,
    });
  } catch (error: any) {
    console.error("Agent execution error:", error);
    return Response.json(
      {
        success: false,
        error: error.message || "Failed to execute agent",
      },
      { status: 500 }
    );
  }
}
