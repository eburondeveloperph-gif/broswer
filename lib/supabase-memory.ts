interface SupabaseMemoryConfig {
  url: string;
  key: string;
  table: string;
}

export interface SessionMemoryItem {
  task: string;
  response: string | null;
  model_role: string;
  model_name: string;
  llm_provider: string;
  step_count: number | null;
  created_at: string;
}

export interface SaveSessionMemoryInput {
  sessionId: string;
  task: string;
  response: string;
  modelRole: string;
  modelName: string;
  llmProvider: string;
  stepCount: number;
  success: boolean;
}

function getSupabaseMemoryConfig(): SupabaseMemoryConfig | null {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;
  const table = process.env.SUPABASE_MEMORY_TABLE || "agent_memory";

  if (!url || !key) return null;
  return { url: url.replace(/\/+$/, ""), key, table };
}

function getSupabaseHeaders(key: string) {
  return {
    apikey: key,
    Authorization: `Bearer ${key}`,
    "Content-Type": "application/json",
  };
}

export async function loadRecentSessionMemory(
  sessionId: string,
  limit = 6
): Promise<SessionMemoryItem[]> {
  const config = getSupabaseMemoryConfig();
  if (!config) return [];

  try {
    const url = new URL(`${config.url}/rest/v1/${config.table}`);
    url.searchParams.set(
      "select",
      "task,response,model_role,model_name,llm_provider,step_count,created_at"
    );
    url.searchParams.set("session_id", `eq.${sessionId}`);
    url.searchParams.set("order", "created_at.desc");
    url.searchParams.set("limit", String(limit));

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: getSupabaseHeaders(config.key),
      cache: "no-store",
    });

    if (!response.ok) {
      console.warn("Failed to load Supabase memory:", await response.text());
      return [];
    }

    const rows = (await response.json()) as SessionMemoryItem[];
    return Array.isArray(rows) ? rows : [];
  } catch (error) {
    console.warn("Error loading Supabase memory:", error);
    return [];
  }
}

export async function saveSessionMemory(input: SaveSessionMemoryInput) {
  const config = getSupabaseMemoryConfig();
  if (!config) return;

  try {
    const response = await fetch(`${config.url}/rest/v1/${config.table}`, {
      method: "POST",
      headers: {
        ...getSupabaseHeaders(config.key),
        Prefer: "return=minimal",
      },
      body: JSON.stringify({
        session_id: input.sessionId,
        task: input.task,
        response: input.response,
        model_role: input.modelRole,
        model_name: input.modelName,
        llm_provider: input.llmProvider,
        step_count: input.stepCount,
        success: input.success,
      }),
      cache: "no-store",
    });

    if (!response.ok) {
      console.warn("Failed to save Supabase memory:", await response.text());
    }
  } catch (error) {
    console.warn("Error saving Supabase memory:", error);
  }
}

export function buildMemoryPromptContext(items: SessionMemoryItem[]) {
  if (!items.length) return "";

  const ordered = [...items].reverse();
  return ordered
    .map((item, index) => {
      const memoryResponse = (item.response || "").slice(0, 300);
      return `Memory ${index + 1}:
- Task: ${item.task}
- Model: ${item.model_name} (${item.model_role})
- Response summary: ${memoryResponse}`;
    })
    .join("\n\n");
}
