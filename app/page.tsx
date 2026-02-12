"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  ListTree,
  Mic,
  X,
  Pause,
  Square,
  MousePointer2,
} from "lucide-react";
import { useRef, useState } from "react";
import { StepsOverlay } from "@/components/StepsOverlay";

interface BrowserSession {
  sessionId: string;
  liveViewUrl: string;
  cdpWsUrl: string;
  spinUpTime: number;
  provider?: "kernel" | "vps";
}

interface ExecutedCode {
  code: string;
  success: boolean;
  result?: any;
  error?: string;
}

interface StepContentItem {
  type: "tool-call" | "tool-result" | "text";
  toolCallId?: string;
  toolName?: string;
  code?: string;
  result?: any;
  success?: boolean;
  text?: string;
}

interface DetailedStep {
  stepNumber: number;
  finishReason: string | null;
  content: StepContentItem[];
}

interface AutomationResult {
  success: boolean;
  response?: string;
  executedCodes?: ExecutedCode[];
  detailedSteps?: DetailedStep[];
  stepCount?: number;
  timestamp: number;
  error?: string;
  task?: string;
}

export default function HomePage() {
  const createBrowserRequestInFlight = useRef(false);
  const [creatingBrowser, setCreatingBrowser] = useState(false);
  const [runningAutomation, setRunningAutomation] = useState(false);
  const [closingBrowser, setClosingBrowser] = useState(false);
  const [browserSession, setBrowserSession] = useState<BrowserSession | null>(
    null
  );
  const [automationResults, setAutomationResults] = useState<
    AutomationResult[]
  >([]);
  const [error, setError] = useState<string | null>(null);
  const [deployUrl, setDeployUrl] = useState<string | null>(null);
  const [task, setTask] = useState("Go to https://eburon.ai/ and create a blog post");
  const [stepsOverlayResult, setStepsOverlayResult] = useState<AutomationResult | null>(null);

  const totalRuns = automationResults.length;
  const successfulRuns = automationResults.filter((result) => result.success).length;
  const lastResult = automationResults[0];
  const lastStepCount = lastResult?.stepCount ?? 0;
  const lastClickCount =
    lastResult?.detailedSteps?.reduce(
      (sum, step) =>
        sum +
        step.content.filter((item: any) => item.type === "tool-call").length,
      0
    ) ?? 0;

  const createBrowser = async () => {
    if (
      createBrowserRequestInFlight.current ||
      creatingBrowser ||
      !!browserSession
    ) {
      return;
    }

    createBrowserRequestInFlight.current = true;
    setCreatingBrowser(true);
    setError(null);
    setDeployUrl(null);

    try {
      const response = await fetch("/api/create-browser", {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        setBrowserSession({
          sessionId: data.sessionId,
          liveViewUrl: data.liveViewUrl,
          cdpWsUrl: data.cdpWsUrl,
          spinUpTime: data.spinUpTime,
          provider: data.provider,
        });
      } else {
        if (data.error === "MISSING_API_KEY" && data.deployUrl) {
          setDeployUrl(data.deployUrl);
        }
        setError(data.message || data.error || "Failed to create browser");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to connect to API"
      );
    } finally {
      setCreatingBrowser(false);
      createBrowserRequestInFlight.current = false;
    }
  };

  const runAutomation = async () => {
    if (!browserSession || !task.trim()) return;

    setRunningAutomation(true);

    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sessionId: browserSession.sessionId,
          cdpWsUrl: browserSession.cdpWsUrl,
          provider: browserSession.provider,
          task: task.trim(),
        }),
      });

      const data = await response.json();

      const result: AutomationResult = {
        success: data.success,
        response: data.response,
        executedCodes: data.executedCodes,
        detailedSteps: data.detailedSteps,
        stepCount: data.stepCount,
        error: data.error,
        task: task.trim(),
        timestamp: Date.now(),
      };

      setAutomationResults((prev) => [result, ...prev]);

      // Clear the task input after successful execution
      if (data.success) {
        setTask("");
      }
    } catch (err) {
      const result: AutomationResult = {
        success: false,
        error: "Failed to run Eburon agent",
        task: task.trim(),
        timestamp: Date.now(),
      };
      setAutomationResults((prev) => [result, ...prev]);
    } finally {
      setRunningAutomation(false);
    }
  };

  const closeBrowser = async () => {
    if (!browserSession) return;

    setClosingBrowser(true);

    try {
      const response = await fetch("/api/delete-browser", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sessionId: browserSession.sessionId,
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Clear browser session and reset state
        setBrowserSession(null);
        setAutomationResults([]);
        setTask("Go to https://eburon.ai/ and create a blog post");
      } else {
        setError(data.error || "Failed to close browser");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to close browser"
      );
    } finally {
      setClosingBrowser(false);
    }
  };

  return (
    <div className="min-h-screen relative flex flex-col">
      {/* Radial Glow Effect */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[600px] bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))]"></div>
      </div>

      {/* Main Content */}
      <section
        className={`relative z-10 flex-grow ${
          browserSession
            ? "py-12 lg:py-20"
            : "py-8 sm:py-10 flex items-center"
        }`}
      >
        <div
          className={`container mx-auto px-4 ${
            browserSession ? "max-w-5xl" : "max-w-3xl"
          }`}
        >
          <div className={`space-y-8 ${!browserSession ? "text-center" : ""}`}>
            {/* Top description (hide once sandbox is active) */}
            {!browserSession && (
              <div className="space-y-2">
                <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
                  Eburon Autonomous Agent
                </h1>
                <p className="text-sm sm:text-base text-gray-400 max-w-xl mx-auto">
                  Launch a disposable sandbox browser and run autonomous web simulations in seconds.
                </p>
              </div>
            )}

            {/* Simulation summary (clicks, runs, steps) */}
            {automationResults.length > 0 && (
              <Card>
                <CardContent className="py-4">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs sm:text-sm">
                    <div className="space-y-1">
                      <p className="text-muted-foreground">Total simulations</p>
                      <p className="font-mono text-base sm:text-lg font-semibold">
                        {totalRuns}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-muted-foreground">Successful</p>
                      <p className="font-mono text-base sm:text-lg font-semibold">
                        {successfulRuns}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-muted-foreground">Last run steps</p>
                      <p className="font-mono text-base sm:text-lg font-semibold">
                        {lastStepCount}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-muted-foreground">Last run clicks</p>
                      <p className="font-mono text-base sm:text-lg font-semibold">
                        {lastClickCount}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <div className="space-y-8">
              {/* Step 1: Create Browser / Sandbox */}
              {!browserSession && (
                <div className="space-y-3 text-center">
                  <Button
                    variant="vercel"
                    size="lg"
                    onClick={createBrowser}
                    disabled={creatingBrowser}
                    className="text-base px-8 py-6 h-auto font-semibold w-full sm:w-auto"
                  >
                    {creatingBrowser ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Creating Sandbox Browser...
                      </>
                    ) : (
                      <>
                        Create Sandbox Browser
                        <span className="ml-1">→</span>
                      </>
                    )}
                  </Button>
                  <p className="text-xs text-gray-600">
                    Creates an isolated browser sandbox you can freely automate.
                  </p>
                  <div className="mt-6 flex justify-center">
                    <div className="relative w-full max-w-md aspect-video rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-950 to-black shadow-[0_0_40px_rgba(0,0,0,0.6)] overflow-hidden">
                      <div className="absolute inset-3 rounded-xl border border-slate-700/60 bg-slate-900/90 overflow-hidden">
                        <div className="absolute inset-0 sandbox-demo-stage">
                          <div className="sandbox-demo-window sandbox-demo-main absolute left-3 right-3 top-3 rounded-lg border border-slate-600/70 bg-slate-900/95 overflow-hidden">
                            <div className="h-6 border-b border-slate-700/70 bg-slate-800/95 px-2 flex items-center gap-1.5">
                              <span className="h-2 w-2 rounded-full bg-red-400/80" />
                              <span className="h-2 w-2 rounded-full bg-yellow-400/80" />
                              <span className="h-2 w-2 rounded-full bg-emerald-400/80" />
                              <span className="ml-2 h-3 w-20 rounded-full bg-slate-700/90" />
                            </div>
                            <div className="p-2 space-y-1.5">
                              <div className="h-2.5 rounded bg-slate-700/80" />
                              <div className="h-2.5 w-4/5 rounded bg-slate-800/80" />
                              <div className="h-12 rounded-md bg-slate-800/60" />
                              <div className="grid grid-cols-3 gap-1.5">
                                <div className="h-8 rounded bg-slate-800/70" />
                                <div className="h-8 rounded bg-slate-800/60" />
                                <div className="h-8 rounded bg-slate-800/50" />
                              </div>
                            </div>
                          </div>

                          <div className="sandbox-demo-window sandbox-demo-popup absolute right-5 top-8 w-32 sm:w-36 rounded-md border border-cyan-400/30 bg-slate-900/95 overflow-hidden shadow-lg shadow-cyan-500/20">
                            <div className="h-5 border-b border-slate-700/70 bg-slate-800/95 px-2 flex items-center">
                              <span className="h-2 w-8 rounded-full bg-cyan-300/60" />
                            </div>
                            <div className="p-2 space-y-1">
                              <div className="h-2 rounded bg-slate-700/80" />
                              <div className="h-2 w-5/6 rounded bg-slate-800/70" />
                              <div className="h-8 rounded bg-slate-800/60" />
                            </div>
                          </div>

                          <div className="absolute left-3 right-3 bottom-2 h-8 rounded-md border border-slate-700/70 bg-black/50 px-2 flex items-center gap-2">
                            <div className="sandbox-demo-dock-icon h-5 w-5 rounded bg-cyan-400/70 shadow-md shadow-cyan-400/30" />
                            <div className="h-5 w-5 rounded bg-violet-400/60" />
                            <div className="h-5 w-5 rounded bg-emerald-400/60" />
                          </div>

                          <div className="sandbox-demo-cursor pointer-events-none absolute">
                            <MousePointer2 className="h-4 w-4 text-white drop-shadow-lg" />
                            <span className="sandbox-demo-click absolute left-3 top-3 h-2.5 w-2.5 rounded-full border border-cyan-300/80" />
                          </div>
                        </div>
                      </div>

                      <div className="absolute -bottom-4 left-1/2 h-4 w-20 -translate-x-1/2 rounded-b-3xl bg-slate-800" />
                    </div>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && !browserSession && (
                <Card className="text-left max-w-xl mx-auto">
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <XCircle className="w-5 h-5 text-red-600" />
                        <span className="font-semibold">Error</span>
                      </div>
                      <p className="font-mono text-sm text-red-600">{error}</p>
                      {deployUrl && (
                        <div className="pt-2">
                          <p className="text-sm text-muted-foreground mb-3">
                            Deploy this template with the Maximus integration to get started:
                          </p>
                          <a
                            href={deployUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-block"
                          >
                            <img
                              src="https://vercel.com/button"
                              alt="Deploy with Vercel"
                            />
                          </a>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Live View and Automation Controls */}
              {browserSession && (
                <div className="space-y-6">
                  {/* Live View */}
                  <Card>
                    <CardContent className="p-3 sm:p-4">
                      <div className="space-y-4">
                        <div className="rounded-lg overflow-hidden border bg-black">
                          <div className="relative w-full aspect-video sm:aspect-[16/10] min-h-[220px] max-h-[68vh] sm:max-h-[75vh]">
                            <iframe
                              src={browserSession.liveViewUrl}
                              title="Eburon browser live view"
                              className="absolute inset-0 block w-full h-full"
                              allow="camera; microphone; display-capture"
                            />
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Step 2: Run Eburon Agent */}
                  <Card>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="space-y-2 text-left">
                          <label
                            htmlFor="task-input"
                            className="text-sm font-medium"
                          >
                            Describe what you want the browser to do
                          </label>
                          <div className="relative">
                            <Textarea
                              id="task-input"
                              value={task}
                              onChange={(e) => setTask(e.target.value)}
                              placeholder={
                                automationResults.length > 0
                                  ? "Enter next task for the browser agent"
                                : "Go to https://eburon.ai/ and create a blog post"
                              }
                              disabled={runningAutomation}
                              className="min-h-[100px] resize-none placeholder:text-gray-600 pr-10"
                              onKeyDown={(e) => {
                                if (
                                  e.key === "Enter" &&
                                  (e.metaKey || e.ctrlKey)
                                ) {
                                  e.preventDefault();
                                  runAutomation();
                                }
                              }}
                            />
                            <button
                              type="button"
                              className="absolute right-2 bottom-2 inline-flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-black/60 text-gray-200 hover:bg-black/80"
                              aria-label="Dictate task with microphone"
                              onClick={() => {
                                // Voice input capture can be wired here.
                              }}
                            >
                              <Mic className="w-3 h-3" />
                            </button>
                          </div>
                          <p className="hidden sm:block text-xs text-muted-foreground">
                            Press Cmd/Ctrl + Enter to run
                          </p>
                        </div>
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                          <Button
                            size="lg"
                            onClick={runAutomation}
                            disabled={runningAutomation || !task.trim()}
                            className="w-full sm:flex-1 text-lg py-6"
                          >
                            {runningAutomation ? (
                              <>
                                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                                Eburon Agent Running...
                              </>
                            ) : (
                              "Run Eburon Agent"
                            )}
                          </Button>
                          <div className="grid grid-cols-3 gap-2 sm:flex sm:items-center sm:gap-2 sm:flex-shrink-0">
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              onClick={closeBrowser}
                              disabled={closingBrowser || runningAutomation}
                              className="h-12 w-full sm:w-12"
                              aria-label="Close browser sandbox"
                              title="Close browser sandbox"
                            >
                              {closingBrowser ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <X className="h-4 w-4" />
                              )}
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              className="h-12 w-full sm:w-12"
                              aria-label="Pause simulation"
                              title="Pause simulation"
                            >
                              <Pause className="h-4 w-4" />
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              className="h-12 w-full sm:w-12"
                              aria-label="Stop simulation"
                              title="Stop simulation"
                            >
                              <Square className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Automation Results */}
                  {automationResults.length > 0 && (
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold text-left">
                        Agent Results
                      </h3>
                      {automationResults.map((result, index) => (
                        <Card key={result.timestamp} className="text-left">
                          <CardContent>
                            <div className="space-y-4">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  {result.success ? (
                                    <>
                                      <CheckCircle2 className="w-5 h-5 text-green-600" />
                                      <span className="font-semibold">
                                        Run #{automationResults.length - index}
                                      </span>
                                    </>
                                  ) : (
                                    <>
                                      <XCircle className="w-5 h-5 text-red-600" />
                                      <span className="font-semibold">
                                        Run #{automationResults.length - index}{" "}
                                        Failed
                                      </span>
                                    </>
                                  )}
                                </div>
                                <span className="text-xs text-muted-foreground">
                                  {new Date(result.timestamp).toLocaleTimeString()}
                                </span>
                              </div>

                              {/* Task Description */}
                              {result.task && (
                                <div className="p-3 bg-muted rounded-md">
                                  <p className="text-sm text-muted-foreground mb-1">
                                    Task:
                                  </p>
                                  <p className="text-sm">{result.task}</p>
                                </div>
                              )}

                              {result.success ? (
                                <div className="space-y-3">
                                  {/* Agent Response */}
                                  {result.response && (
                                    <div>
                                      <p className="text-sm text-muted-foreground mb-1">
                                        Response:
                                      </p>
                                      <p className="text-sm">{result.response}</p>
                                    </div>
                                  )}

                                  {/* Step Count and View Steps Button */}
                                  {result.stepCount !== undefined && (
                                    <div className="flex items-center gap-2">
                                      <Badge variant="secondary">
                                        {result.stepCount} steps
                                      </Badge>
                                      {result.detailedSteps &&
                                        result.detailedSteps.length > 0 && (
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setStepsOverlayResult(result)}
                                            className="ml-2"
                                          >
                                            <ListTree className="w-4 h-4 mr-1" />
                                            View Steps
                                          </Button>
                                        )}
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <div className="space-y-2">
                                  <div>
                                    <span className="text-sm text-muted-foreground">
                                      Error:
                                    </span>
                                    <p className="font-mono text-sm text-red-600">
                                      {result.error}
                                    </p>
                                  </div>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="pb-8 mt-auto">
        <div className="container mx-auto px-4 text-center text-gray-600 text-xs">
          <p>
            Powered by{" "}
            <a
              href="https://eburon.ai"
              className="font-medium hover:text-gray-400 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Eburon AI
            </a>
          </p>
        </div>
      </footer>

      <style jsx>{`
        .sandbox-demo-stage {
          background: radial-gradient(
            ellipse at top,
            rgba(59, 130, 246, 0.08),
            rgba(15, 23, 42, 0.95)
          );
        }

        .sandbox-demo-window {
          transform-origin: top right;
        }

        .sandbox-demo-main {
          animation: sandbox-main-window 6.8s ease-in-out infinite;
        }

        .sandbox-demo-popup {
          animation: sandbox-popup-window 6.8s ease-in-out infinite;
        }

        .sandbox-demo-dock-icon {
          animation: sandbox-dock-pulse 6.8s ease-in-out infinite;
        }

        .sandbox-demo-cursor {
          animation: sandbox-cursor-move 6.8s linear infinite;
        }

        .sandbox-demo-click {
          animation: sandbox-click-pulse 6.8s ease-out infinite;
        }

        @keyframes sandbox-main-window {
          0%,
          15% {
            transform: translateY(4px) scale(0.98);
            opacity: 0.9;
          }
          25%,
          55% {
            transform: translateY(0) scale(1);
            opacity: 1;
          }
          65%,
          100% {
            transform: translateY(2px) scale(0.99);
            opacity: 0.95;
          }
        }

        @keyframes sandbox-popup-window {
          0%,
          12% {
            transform: translate(84px, -24px) scale(0.7);
            opacity: 0;
          }
          22%,
          50% {
            transform: translate(0, 0) scale(1);
            opacity: 1;
          }
          62%,
          76% {
            transform: translate(70px, 92px) scale(0.2);
            opacity: 0.6;
          }
          80%,
          100% {
            transform: translate(84px, 106px) scale(0.08);
            opacity: 0;
          }
        }

        @keyframes sandbox-dock-pulse {
          0%,
          58%,
          100% {
            transform: scale(1);
            box-shadow: 0 0 0 rgba(34, 211, 238, 0);
          }
          68% {
            transform: scale(1.08);
            box-shadow: 0 0 14px rgba(34, 211, 238, 0.45);
          }
        }

        @keyframes sandbox-cursor-move {
          0% {
            transform: translate(24px, 24px);
          }
          20% {
            transform: translate(120px, 46px);
          }
          40% {
            transform: translate(145px, 80px);
          }
          58% {
            transform: translate(206px, 138px);
          }
          78% {
            transform: translate(54px, 112px);
          }
          100% {
            transform: translate(24px, 24px);
          }
        }

        @keyframes sandbox-click-pulse {
          0%,
          18%,
          38%,
          56%,
          100% {
            transform: scale(0.45);
            opacity: 0;
          }
          6%,
          26%,
          46%,
          66% {
            transform: scale(1.9);
            opacity: 0.95;
          }
          12%,
          32%,
          52%,
          72% {
            transform: scale(2.4);
            opacity: 0;
          }
        }
      `}</style>

      {/* Steps Overlay */}
      {stepsOverlayResult && (
        <StepsOverlay
          open={!!stepsOverlayResult}
          onOpenChange={(open) => !open && setStepsOverlayResult(null)}
          steps={stepsOverlayResult.detailedSteps || []}
          task={stepsOverlayResult.task || ""}
        />
      )}
    </div>
  );
}