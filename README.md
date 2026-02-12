# Kernel + Vercel Template

Next.js + Kernel template for running AI-powered browser automations with natural language on Vercel.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fonkernel%2Fkernel-nextjs-template&env=OLLAMA_API_KEY&project-name=kernel-nextjs-template&repository-name=kernel-nextjs-template&products=%5B%7B%22type%22%3A%22integration%22%2C%22integrationSlug%22%3A%22kernel%22%2C%22productSlug%22%3A%22kernel%22%2C%22protocol%22%3A%22other%22%7D%5D)

## Overview

This template shows how to:

- Create serverless browsers with live view using the Kernel SDK
- Optionally run sandbox sessions from your own VPS via CDP + live view URL
- Describe browser tasks in natural language
- Route each task to one of three agents/models:
  - Automation agent: `kimi-k2-thinking:cloud`
  - Reasoning agent: `deepseek-v3.1:671b-cloud`
  - OCR agent: `glm-ocr:latest`
- Use an AI agent to execute browser automation code via AI SDK tools in Next.js API routes
- Display live browser view and automation results in a modern Next.js UI

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS v4
- **UI Components**: shadcn/ui
- **AI**: Vercel AI SDK with Ollama (`kimi-k2-thinking:cloud`)
- **Browser Automation**: Kernel SDK + Kernel AI SDK (`@onkernel/ai-sdk`)
- **Package Manager**: Bun
- **Deployment**: Vercel

## Getting Started

### Prerequisites

- Node.js 18+
- [Bun](https://bun.sh) (package manager)
- A Kernel account and API key
- Ollama running locally on `http://localhost:11434`
- [Supabase CLI](https://supabase.com/docs/guides/local-development/cli/getting-started) (optional, for local memory)
- Vercel account (optional, for deployment)

### Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd nextjs-kernel-template
   ```

2. **Install dependencies**:

   ```bash
   bun install
   ```

3. **Set up Kernel**:

   Get your Kernel API key from one of these sources:

   - **Option 1 (Recommended)**: Install the [Kernel integration](https://vercel.com/integrations/kernel) from the Vercel Marketplace
   - **Option 2**: Get your API key from [https://dashboard.onkernel.com](https://dashboard.onkernel.com)

4. **Configure environment variables**:

   Create a `.env` file:

   ```bash
   touch .env.local
   ```

   Add your API keys:

   ```dotenv
   KERNEL_API_KEY=your_kernel_api_key_here
   SANDBOX_PROVIDER=kernel

   # Local Ollama (default path)
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_API_KEY=ollama
   OLLAMA_MODEL=kimi-k2-thinking:cloud
   OLLAMA_MODEL_AUTOMATION=kimi-k2-thinking:cloud
   OLLAMA_MODEL_REASONING=deepseek-v3.1:671b-cloud
   OLLAMA_MODEL_OCR=glm-ocr:latest

   # Optional cloud fallback (used automatically if local Ollama is unreachable)
   OLLAMA_CLOUD_BASE_URL=
   OLLAMA_CLOUD_API_KEY=
   OLLAMA_CLOUD_MODEL=
   OLLAMA_CLOUD_MODEL_AUTOMATION=kimi-k2-thinking:cloud
   OLLAMA_CLOUD_MODEL_REASONING=deepseek-v3.1:671b-cloud
   OLLAMA_CLOUD_MODEL_OCR=glm-ocr:latest
   OLLAMA_CLOUD_DEPLOYMENT_SHA=
   OLLAMA_CLOUD_SSH_PUBLIC_KEY=

   # Supabase memory (self-hosted local or cloud)
   SUPABASE_URL=
   SUPABASE_ANON_KEY=
   SUPABASE_SERVICE_ROLE_KEY=
   SUPABASE_MEMORY_TABLE=agent_memory

   # Optional VPS sandbox mode
   # Set SANDBOX_PROVIDER=vps to use your VPS browser instead of Kernel.
   VPS_SANDBOX_SESSION_ID=vps-sandbox-session
   VPS_SANDBOX_LIVE_VIEW_URL=
   VPS_SANDBOX_CDP_WS_URL=
   ```

5. **Optional: start self-hosted Supabase locally**:

   ```bash
   bun run supabase:start
   supabase db query < supabase/agent-memory.sql
   ```

   Then set local memory vars in `.env.local` (typical local URL: `http://127.0.0.1:54321`).

6. **Run the development server**:

   ```bash
   bun dev
   ```

7. **Open** [http://localhost:3000](http://localhost:3000) in your browser

## How It Works

1. **Create Browser**: Click "Create Browser" to provision a serverless Kernel browser with live view capabilities
2. **Describe Your Task**: Enter what you want the browser to do in natural language (e.g., "Go to Hacker News and get the top article title")
3. **Watch AI Execute**: The AI agent interprets your task and uses Kernel's AI SDK-compatible browser automation tool to execute it in real-time
4. **View Results**: See the agent's response, step count, and click "View Steps" to inspect the generated code and execution details

## Code Structure

```text
app/
├── api/
│   ├── agent/
│   │   └── route.ts          # AI agent endpoint with browser automation tool
│   ├── create-browser/
│   │   └── route.ts          # Creates a serverless Kernel browser
│   └── delete-browser/
│       └── route.ts          # Closes browser session
├── page.tsx                  # Main UI with live view and controls
├── layout.tsx                # Root layout
└── globals.css               # Global styles

components/
├── Header.tsx                # App header with branding
├── StepsOverlay.tsx          # Modal showing agent execution steps
└── ui/                       # shadcn/ui components
    ├── button.tsx
    ├── card.tsx
    ├── textarea.tsx
    └── ...

lib/
└── utils.ts                  # Utility functions
```

### Key Code Example

**Step 1: Create Browser** (`app/api/create-browser/route.ts`)

```typescript
import { Kernel } from "@onkernel/sdk";

// Initialize Kernel client
const kernel = new Kernel({ apiKey: process.env.KERNEL_API_KEY });

// Create a serverless browser with live view
const browser = await kernel.browsers.create({
  stealth: true,
  headless: false,
});

// Return browser details to client
return {
  sessionId: browser.session_id,
  liveViewUrl: browser.browser_live_view_url,
  cdpWsUrl: browser.cdp_ws_url,
};
```

**Step 2: Run AI Agent** (`app/api/agent/route.ts`)

```typescript
import { createOpenAI } from "@ai-sdk/openai";
import { playwrightExecuteTool } from "@onkernel/ai-sdk";
import { Kernel } from "@onkernel/sdk";
import { Experimental_Agent as Agent, stepCountIs } from "ai";

// Initialize Kernel instance
const kernel = new Kernel({ apiKey: process.env.KERNEL_API_KEY });

const ollama = createOpenAI({
  baseURL: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1",
  apiKey: process.env.OLLAMA_API_KEY ?? "ollama",
  name: "ollama",
});

// Initialize the AI agent with an Ollama model and Kernel's AI SDK-compatible browser automation tool
const agent = new Agent({
  model: ollama(process.env.OLLAMA_MODEL ?? "kimi-k2-thinking:cloud"),
  tools: {
    playwright_execute: playwrightExecuteTool({
      client: kernel,
      sessionId: sessionId,
    }),
  },
  stopWhen: stepCountIs(20),
  system: `You are a browser automation expert with access to a Playwright execution tool...`,
});

// Execute the agent with the user's task
const { text, steps } = await agent.generate({
  prompt: task, // e.g., "Go to news.ycombinator.com and get the first article title"
});
```

## Deployment

### Deploy to Vercel

1. **Push to GitHub**

2. **Connect to Vercel**:

   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Add your environment variables (`KERNEL_API_KEY`, local Ollama vars, and optional `OLLAMA_CLOUD_*` fallback vars)
   - Deploy!

3. **Using Vercel Marketplace Integration**:
   - Install [Kernel from Vercel Marketplace](https://vercel.com/integrations/kernel)
   - The integration will automatically add the Kernel API key to your project
   - Add your Ollama variables manually
   - Deploy your project

### Environment Variables

Make sure to add these environment variables in your Vercel project settings:

- `KERNEL_API_KEY` - Your Kernel API key
- `SANDBOX_PROVIDER` - `kernel` (default) or `vps`
- `OLLAMA_BASE_URL` - Local Ollama base URL (default: `http://localhost:11434/v1`)
- `OLLAMA_API_KEY` - Local Ollama API key (`ollama` is fine for localhost)
- `OLLAMA_MODEL` - Backward-compatible default model
- `OLLAMA_MODEL_AUTOMATION` - Model used for web automation tasks
- `OLLAMA_MODEL_REASONING` - Model used for analysis/planning tasks
- `OLLAMA_MODEL_OCR` - Model used for OCR/vision extraction tasks
- `OLLAMA_CLOUD_BASE_URL` - Optional cloud fallback base URL
- `OLLAMA_CLOUD_API_KEY` - Optional cloud fallback API key
- `OLLAMA_CLOUD_MODEL` - Backward-compatible cloud fallback model
- `OLLAMA_CLOUD_MODEL_AUTOMATION` - Optional cloud model override for automation
- `OLLAMA_CLOUD_MODEL_REASONING` - Optional cloud model override for reasoning
- `OLLAMA_CLOUD_MODEL_OCR` - Optional cloud model override for OCR
- `OLLAMA_CLOUD_DEPLOYMENT_SHA` - Optional header value sent as `x-deployment-sha` for cloud routing
- `OLLAMA_CLOUD_SSH_PUBLIC_KEY` - Optional header value sent as `x-ssh-public-key` for SSH-bound cloud routing
- `SUPABASE_URL` - Supabase REST base URL (`http://127.0.0.1:54321` for local self-hosting)
- `SUPABASE_ANON_KEY` - Supabase anon key (optional when service role key is set)
- `SUPABASE_SERVICE_ROLE_KEY` - Preferred key for server-side memory writes
- `SUPABASE_MEMORY_TABLE` - Memory table name (default: `agent_memory`)
- `VPS_SANDBOX_SESSION_ID` - Session label when VPS mode is enabled
- `VPS_SANDBOX_LIVE_VIEW_URL` - Live-view iframe URL hosted from your VPS
- `VPS_SANDBOX_CDP_WS_URL` - CDP websocket URL for your VPS browser instance

## Learn More

- [Kernel Documentation](https://docs.onkernel.com)
- [Kernel AI SDK](https://www.onkernel.com/docs/integrations/vercel/ai-sdk)
- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Next.js Documentation](https://nextjs.org/docs)

---

Built with [Kernel](https://dashboard.onkernel.com), [Vercel AI SDK](https://sdk.vercel.ai), and [Vercel](https://vercel.com)
