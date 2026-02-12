import { createOpenAI } from "@ai-sdk/openai";
import { playwrightExecuteTool } from "@onkernel/ai-sdk";
import { Kernel } from "@onkernel/sdk";
import { Experimental_Agent as Agent, stepCountIs, tool } from "ai";
import { z } from "zod";

export const maxDuration = 300; // 5 minutes timeout for long-running agent operations

async function normalizeToSinglePage(kernel: Kernel, sessionId: string) {
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

export async function POST(req: Request) {
  try {
    const { sessionId, task, serverTarget } = await req.json();

    if (!sessionId || !task) {
      return Response.json(
        { error: "Missing sessionId or task" },
        { status: 400 }
      );
    }

    const apiKey = process.env.KERNEL_API_KEY;
    const requestedServerTarget =
      serverTarget === "vps" ? "vps" : "cloud-eu";

    let ollamaBaseUrl: string;
    let ollamaApiKey: string;
    let ollamaModel: string;

    if (requestedServerTarget === "cloud-eu") {
      const cloudBaseUrl = process.env.OLLAMA_CLOUD_BASE_URL;
      const cloudApiKey = process.env.OLLAMA_CLOUD_API_KEY;
      if (!cloudBaseUrl || !cloudApiKey) {
        return Response.json(
          {
            success: false,
            error:
              "Cloud Server EU selected but OLLAMA_CLOUD_BASE_URL and/or OLLAMA_CLOUD_API_KEY are not set. Add these env vars or use VPS Self-Hosted.",
          },
          { status: 400 }
        );
      }
      ollamaBaseUrl = cloudBaseUrl;
      ollamaApiKey = cloudApiKey || "ollama";

      const cloudGeneralModel =
        process.env.OLLAMA_CLOUD_MODEL || "gpt-oss:120b-cloud";
      const cloudCodingModel =
        process.env.OLLAMA_CLOUD_MODEL_CODING || "kimi-k2-thinking:cloud";
      const cloudReasoningModel =
        process.env.OLLAMA_CLOUD_MODEL_REASONING || "deepseek-v3.1:671b-cloud";
      const cloudOcrModel =
        process.env.OLLAMA_CLOUD_MODEL_OCR || "glm-ocr:latest";

      const lowerTask = task.toLowerCase();
      if (/(ocr|extract\s+text|read\s+text|image\s+text|scan)/.test(lowerTask)) {
        ollamaModel = cloudOcrModel;
      } else if (/(code|coding|typescript|javascript|python|refactor|debug|implement|bug\s*fix)/.test(lowerTask)) {
        ollamaModel = cloudCodingModel;
      } else if (/(reason|analy|plan|architecture|strategy|compare)/.test(lowerTask)) {
        ollamaModel = cloudReasoningModel;
      } else {
        ollamaModel = cloudGeneralModel;
      }
    } else {
      ollamaBaseUrl =
        process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1";
      ollamaApiKey = process.env.OLLAMA_API_KEY || "ollama";
      ollamaModel = process.env.OLLAMA_MODEL || "eburon-new/eburon-2.3:latest";
    }

    if (!apiKey) {
      return Response.json(
        { error: "KERNEL_API_KEY environment variable is not set" },
        { status: 400 }
      );
    }

    const kernel = new Kernel({ apiKey });
    await normalizeToSinglePage(kernel, sessionId);
    const ollama = createOpenAI({
      baseURL: ollamaBaseUrl,
      apiKey: ollamaApiKey || "ollama",
      name: "ollama",
    });

    // Initialize the AI agent with an Ollama model
    const agent = new Agent({
      model: ollama.chat(ollamaModel),
      tools: {
        playwright_execute: playwrightExecuteTool({
          client: kernel,
          sessionId: sessionId,
        }),
      },
      stopWhen: stepCountIs(20),
      system: `You are the Eburon Autonomous Agent, a browser automation expert operating inside a disposable, sandboxed environment with access to a Playwright execution tool.

Available tools:
- playwright_execute: Executes JavaScript/Playwright code in the browser. Has access to 'page', 'context', and 'browser' objects. Returns the result of your code.

When given a task:
1. If no URL is provided, FIRST get the current page context:
   return { url: page.url(), title: await page.title() }
2. If a URL is provided, navigate to it using page.goto()
3. Use appropriate selectors (page.locator, page.getByRole, etc.) to interact with elements
4. Safely handle authentication flows when the user explicitly provides credentials (for example, filling login forms), but never attempt to obtain or exfiltrate secrets the user did not clearly request or supply.
5. Always return the requested data from your code execution.
6. Keep the session to a single active browser page/tab unless the user explicitly asks for multiple tabs or popups.
7. Never call context.newPage() or open new windows unless explicitly requested; if a popup/new tab appears, close the extra page and continue on the main page.

Behavior:
- Break complex tasks into small, focused executions rather than writing long scripts.
- After each tool call, clearly describe in natural language what you clicked, typed, or observed so users can understand the simulation steps.
- Prefer reusing the existing page for navigation and interactions to avoid duplicate browser windows.
- Execute tasks autonomously without asking clarifying questions when possible, making reasonable assumptions while respecting security, privacy, and website terms of service.`,
    });

    // Execute the agent with cloud->VPS fallback when cloud auth/quota fails.
    let resolvedServerTarget = requestedServerTarget;
    let text: string;
    let steps: any[];
    let usage: any;

    try {
      ({ text, steps, usage } = await agent.generate({
        prompt: task,
      }));
    } catch (primaryError: any) {
      const message = (primaryError?.message ?? String(primaryError ?? "")).toLowerCase();
      const shouldFallbackToVps =
        requestedServerTarget === "cloud-eu" &&
        /(401|403|429|unauthorized|forbidden|too many requests|usage limit|api key|token|quota|item_reference|unknown input item type)/.test(
          message
        );

      if (!shouldFallbackToVps) {
        throw primaryError;
      }

      const fallbackProvider = createOpenAI({
        baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1",
        apiKey: process.env.OLLAMA_API_KEY || "ollama",
        name: "ollama",
      });
      const fallbackModel =
        process.env.OLLAMA_MODEL || "eburon-new/eburon-2.3:latest";
      const fallbackAgent = new Agent({
        model: fallbackProvider.chat(fallbackModel),
        tools: {
          playwright_execute: playwrightExecuteTool({
            client: kernel,
            sessionId: sessionId,
          }),
        },
        stopWhen: stepCountIs(20),
        system: `You are the Eburon Autonomous Agent, a browser automation expert operating inside a disposable, sandboxed environment with access to a Playwright execution tool.

Available tools:
- playwright_execute: Executes JavaScript/Playwright code in the browser. Has access to 'page', 'context', and 'browser' objects. Returns the result of your code.

When given a task:
1. If no URL is provided, FIRST get the current page context:
   return { url: page.url(), title: await page.title() }
2. If a URL is provided, navigate to it using page.goto()
3. Use appropriate selectors (page.locator, page.getByRole, etc.) to interact with elements
4. Safely handle authentication flows when the user explicitly provides credentials (for example, filling login forms), but never attempt to obtain or exfiltrate secrets the user did not clearly request or supply.
5. Always return the requested data from your code execution.
6. Keep the session to a single active browser page/tab unless the user explicitly asks for multiple tabs or popups.
7. Never call context.newPage() or open new windows unless explicitly requested; if a popup/new tab appears, close the extra page and continue on the main page.

Behavior:
- Break complex tasks into small, focused executions rather than writing long scripts.
- After each tool call, clearly describe in natural language what you clicked, typed, or observed so users can understand the simulation steps.
- Prefer reusing the existing page for navigation and interactions to avoid duplicate browser windows.
- Execute tasks autonomously without asking clarifying questions when possible, making reasonable assumptions while respecting security, privacy, and website terms of service.`,
      });

      ({ text, steps, usage } = await fallbackAgent.generate({
        prompt: task,
      }));
      resolvedServerTarget = "vps";
    }

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

    return Response.json({
      success: true,
      response: text,
      executedCodes,
      detailedSteps,
      stepCount: steps.length,
      usage,
      serverTarget: resolvedServerTarget,
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
