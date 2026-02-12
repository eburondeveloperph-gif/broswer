interface TtsRequestBody {
  text?: string;
  voice?: string;
}

function normalizeBaseUrl(url: string) {
  return url.replace(/\/+$/, "");
}

export async function POST(req: Request) {
  try {
    const { text, voice } = (await req.json()) as TtsRequestBody;
    const inputText = (text || "").trim();

    if (!inputText) {
      return Response.json({ error: "Missing text" }, { status: 400 });
    }

    const voiceServiceBaseUrl = process.env.VOICE_SERVICE_BASE_URL;
    if (voiceServiceBaseUrl) {
      const response = await fetch(
        `${normalizeBaseUrl(voiceServiceBaseUrl)}/tts`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: inputText,
            voice: voice || process.env.VOICE_SERVICE_VOICE || "en-us",
            speed: Number(process.env.VOICE_SERVICE_SPEED || "170"),
          }),
        }
      );

      if (!response.ok) {
        const payload = await response.text();
        return Response.json(
          { error: payload || "TTS failed" },
          { status: response.status }
        );
      }

      const audioBuffer = await response.arrayBuffer();
      return new Response(audioBuffer, {
        status: 200,
        headers: {
          "Content-Type": response.headers.get("content-type") || "audio/wav",
          "Cache-Control": "no-store",
        },
      });
    }

    const openAiApiKey = process.env.OPENAI_API_KEY;
    if (!openAiApiKey) {
      return Response.json(
        {
          error:
            "No TTS provider configured. Set VOICE_SERVICE_BASE_URL or OPENAI_API_KEY.",
        },
        { status: 503 }
      );
    }

    const openAiResponse = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAiApiKey}`,
      },
      body: JSON.stringify({
        model: process.env.OPENAI_TTS_MODEL || "gpt-4o-mini-tts",
        voice: voice || process.env.OPENAI_TTS_VOICE || "alloy",
        input: inputText,
        format: "mp3",
      }),
    });

    if (!openAiResponse.ok) {
      const payload = await openAiResponse.json();
      return Response.json(
        { error: payload?.error?.message || "OpenAI TTS failed" },
        { status: openAiResponse.status }
      );
    }

    const audioBuffer = await openAiResponse.arrayBuffer();
    return new Response(audioBuffer, {
      status: 200,
      headers: {
        "Content-Type": "audio/mpeg",
        "Cache-Control": "no-store",
      },
    });
  } catch (error: any) {
    return Response.json(
      { error: error?.message || "Failed to synthesize audio" },
      { status: 500 }
    );
  }
}
