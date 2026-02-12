create extension if not exists pgcrypto;

create table if not exists public.agent_memory (
  id uuid primary key default gen_random_uuid(),
  session_id text not null,
  task text not null,
  response text,
  model_role text not null,
  model_name text not null,
  llm_provider text not null,
  step_count integer,
  success boolean not null default true,
  created_at timestamptz not null default now()
);

create index if not exists agent_memory_session_created_idx
  on public.agent_memory (session_id, created_at desc);
