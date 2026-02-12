function normalizeBaseUrl(url: string) {
  return url.replace(/\/+$/, "");
}

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const audio = formData.get("audio");

    if (!(audio instanceof File)) {
      return Response.json({ error: "Missing audio file" }, { status: 400 });
    }

    const voiceServiceBaseUrl = process.env.VOICE_SERVICE_BASE_URL;
    if (voiceServiceBaseUrl) {
      const proxyFormData = new FormData();
      proxyFormData.append("audio", audio, audio.name || "audio.webm");

      const response = await fetch(
        `${normalizeBaseUrl(voiceServiceBaseUrl)}/stt`,
        {
          method: "POST",
          body: proxyFormData,
        }
      );

      const payload = await response.json();
      if (!response.ok) {
        return Response.json(
          { error: payload?.detail || payload?.error || "STT failed" },
          { status: response.status }
        );
      }

      return Response.json({ text: payload?.text || "" });
    }

    const openAiApiKey = process.env.OPENAI_API_KEY;
    if (!openAiApiKey) {
      return Response.json(
        {
          error:
            "No STT provider configured. Set VOICE_SERVICE_BASE_URL or OPENAI_API_KEY.",
        },
        { status: 503 }
      );
    }

    const sttForm = new FormData();
    sttForm.append("file", audio, audio.name || "audio.webm");
    sttForm.append(
      "model",
      process.env.OPENAI_STT_MODEL || "gpt-4o-mini-transcribe"
    );

    const openAiResponse = await fetch(
      "https://api.openai.com/v1/audio/transcriptions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${openAiApiKey}`,
        },
        body: sttForm,
      }
    );

    const openAiPayload = await openAiResponse.json();
    if (!openAiResponse.ok) {
      return Response.json(
        { error: openAiPayload?.error?.message || "OpenAI STT failed" },
        { status: openAiResponse.status }
      );
    }

    return Response.json({ text: openAiPayload?.text || "" });
  } catch (error: any) {
    return Response.json(
      { error: error?.message || "Failed to transcribe audio" },
      { status: 500 }
    );
  }
}
