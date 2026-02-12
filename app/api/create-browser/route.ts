import { NextResponse } from "next/server";
import { Kernel, NotFoundError } from "@onkernel/sdk";
import {
  clearLatestBrowserSession,
  getLatestBrowserSession,
  getRecentBrowserSession,
  isCreateBrowserInFlight,
  setCreateBrowserInFlight,
  setLatestBrowserSession,
} from "@/lib/browser-session-registry";
import {
  getVpsSandboxConfig,
  resolveSandboxProvider,
} from "@/lib/sandbox-provider";

async function closeAllActiveSessions(kernel: Kernel) {
  const sessionIds: string[] = [];

  for await (const session of kernel.browsers.list()) {
    if (!session.deleted_at) {
      sessionIds.push(session.session_id);
    }
  }

  for (const sessionId of sessionIds) {
    try {
      await kernel.browsers.deleteByID(sessionId);
    } catch (error) {
      if (!(error instanceof NotFoundError)) {
        console.warn(`Failed to close existing browser session ${sessionId}:`, error);
      }
    }
  }
}

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
    console.warn("Failed to normalize initial browser pages:", error);
  }
}

export async function POST() {
  const sandboxProvider = resolveSandboxProvider();

  if (sandboxProvider === "vps") {
    const vpsConfig = getVpsSandboxConfig();
    if (!vpsConfig) {
      return NextResponse.json(
        {
          success: false,
          error:
            "VPS sandbox is enabled but VPS_SANDBOX_LIVE_VIEW_URL or VPS_SANDBOX_CDP_WS_URL is missing",
        },
        { status: 400 }
      );
    }

    const session = {
      sessionId: vpsConfig.sessionId,
      liveViewUrl: vpsConfig.liveViewUrl,
      cdpWsUrl: vpsConfig.cdpWsUrl,
      spinUpTime: 0,
      provider: "vps" as const,
    };
    setLatestBrowserSession(session);

    return NextResponse.json({
      success: true,
      ...session,
    });
  }

  if (isCreateBrowserInFlight()) {
    const recentSession = getRecentBrowserSession();
    if (recentSession) {
      return NextResponse.json({
        success: true,
        ...recentSession,
        reused: true,
      });
    }

    return NextResponse.json(
      { success: false, error: "Browser creation already in progress" },
      { status: 429 }
    );
  }

  setCreateBrowserInFlight(true);

  try {
    const apiKey = process.env.KERNEL_API_KEY;

    if (!apiKey) {
      return NextResponse.json(
        {
          error: "MISSING_API_KEY",
          message: "KERNEL_API_KEY environment variable is not set",
          deployUrl:
            "https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fonkernel%2Fkernel-nextjs-template&env=OLLAMA_API_KEY&project-name=kernel-nextjs-template&repository-name=kernel-nextjs-template&products=%5B%7B%22type%22%3A%22integration%22%2C%22integrationSlug%22%3A%22kernel%22%2C%22productSlug%22%3A%22kernel%22%2C%22protocol%22%3A%22other%22%7D%5D",
        },
        { status: 400 }
      );
    }

    const startTime = Date.now();

    // Initialize Kernel client
    const kernel = new Kernel({ apiKey });

    // Ensure one sandbox session at a time by cleaning up any previously tracked one.
    const previousSession = getLatestBrowserSession();
    if (previousSession) {
      try {
        await kernel.browsers.deleteByID(previousSession.sessionId);
      } catch (error) {
        // Previous session may already be closed externally; ignore missing sessions.
        if (!(error instanceof NotFoundError)) {
          console.warn("Failed to clean previous browser session:", error);
        }
      } finally {
        clearLatestBrowserSession(previousSession.sessionId);
      }
    }

    // Hard cleanup in provider state as well, so stale sessions do not overlap in live view.
    await closeAllActiveSessions(kernel);

    // Create a headful, stealth browser
    console.log("Creating Kernel browser...");
    const browser = await kernel.browsers.create({
      stealth: true,
      headless: false,
    });
    console.log(`Browser created: ${browser.session_id}`);
    if (!browser.browser_live_view_url) {
      throw new Error("Kernel did not return a live view URL");
    }

    // Some environments can start with multiple tabs/windows; normalize immediately.
    await normalizeToSinglePage(kernel, browser.session_id);

    const spinUpTime = Date.now() - startTime;

    const session = {
      sessionId: browser.session_id,
      liveViewUrl: browser.browser_live_view_url,
      cdpWsUrl: browser.cdp_ws_url,
      spinUpTime,
      provider: "kernel" as const,
    };
    setLatestBrowserSession(session);

    return NextResponse.json({
      success: true,
      ...session,
    });
  } catch (error) {
    console.error("Error creating browser:", error);
    return NextResponse.json(
      {
        error: "Failed to create browser",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  } finally {
    setCreateBrowserInFlight(false);
  }
}
