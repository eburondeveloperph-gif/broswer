export type SandboxProvider = "kernel" | "vps";

export interface VpsSandboxConfig {
  sessionId: string;
  liveViewUrl: string;
  cdpWsUrl: string;
}

export function resolveSandboxProvider(input?: string): SandboxProvider {
  const raw = (input || process.env.SANDBOX_PROVIDER || "kernel").toLowerCase();
  return raw === "vps" ? "vps" : "kernel";
}

export function getVpsSandboxConfig(): VpsSandboxConfig | null {
  const liveViewUrl = process.env.VPS_SANDBOX_LIVE_VIEW_URL;
  const cdpWsUrl = process.env.VPS_SANDBOX_CDP_WS_URL;
  const sessionId = process.env.VPS_SANDBOX_SESSION_ID || "vps-sandbox-session";

  if (!liveViewUrl || !cdpWsUrl) {
    return null;
  }

  return {
    sessionId,
    liveViewUrl,
    cdpWsUrl,
  };
}
