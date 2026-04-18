export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type JsonRecord = Record<string, unknown>;

function parseExtraFields(): JsonRecord {
  const raw = process.env.HF_EXTRA_FIELDS_JSON?.trim();
  if (!raw) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw) as JsonRecord;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (error) {
    throw new Error(
      `HF_EXTRA_FIELDS_JSON is not valid JSON: ${error instanceof Error ? error.message : "unknown error"}`
    );
  }
}

function isJsonRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function tryDecodeBase64Image(value: unknown): Uint8Array | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }

  const normalized = value.startsWith("data:")
    ? value.substring(value.indexOf(",") + 1)
    : value;

  try {
    return Uint8Array.from(Buffer.from(normalized, "base64"));
  } catch {
    return null;
  }
}

function extractImageFromJson(payload: unknown): Uint8Array | null {
  const candidates: unknown[] = [];

  if (Array.isArray(payload)) {
    candidates.push(payload[0]);
  } else if (isJsonRecord(payload)) {
    candidates.push(
      payload.image,
      payload.output,
      payload.result,
      Array.isArray(payload.images) ? payload.images[0] : null,
      Array.isArray(payload.data) ? payload.data[0] : null
    );
  } else {
    return null;
  }

  for (const candidate of candidates) {
    if (isJsonRecord(candidate) && "b64_json" in candidate) {
      const decoded = tryDecodeBase64Image(candidate.b64_json);
      if (decoded) {
        return decoded;
      }
    }

    if (isJsonRecord(candidate) && "image" in candidate) {
      const decoded = tryDecodeBase64Image(candidate.image);
      if (decoded) {
        return decoded;
      }
    }

    const decoded = tryDecodeBase64Image(candidate);
    if (decoded) {
      return decoded;
    }
  }

  return null;
}

function jsonResponse(status: number, error: string, details?: string) {
  return Response.json({ error, details }, { status });
}

function validateEndpoint(endpoint: string): { valid: true } | { valid: false; details: string } {
  const trimmed = endpoint.trim();
  if (!trimmed || trimmed.includes("your-endpoint-url-here")) {
    return {
      valid: false,
      details: "HF_IMAGE_API_URL is still using the placeholder value in frontend/.env.local."
    };
  }

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return {
      valid: false,
      details: "HF_IMAGE_API_URL is not a valid URL."
    };
  }

  const host = url.hostname.toLowerCase();
  const path = url.pathname.toLowerCase();

  if (
    host === "huggingface.co" &&
    (path.includes("/blob/") || path.includes("/resolve/") || path.includes("/tree/"))
  ) {
    return {
      valid: false,
      details:
        "The configured URL is a Hugging Face file or repo page, not an inference endpoint. Use a Space API URL or a dedicated Inference Endpoint URL instead."
    };
  }

  if (host === "huggingface.co" && /^\/[^/]+\/[^/]+\/?$/.test(path)) {
    return {
      valid: false,
      details:
        "The configured URL points to a Hugging Face model repo page. A model page cannot process uploaded images directly."
    };
  }

  return { valid: true };
}

export async function POST(request: Request) {
  const endpoint = process.env.HF_IMAGE_API_URL;
  if (!endpoint) {
    return jsonResponse(
      500,
      "Missing Hugging Face endpoint.",
      "Set HF_IMAGE_API_URL in frontend/.env.local."
    );
  }

  const endpointValidation = validateEndpoint(endpoint);
  if (!endpointValidation.valid) {
    return jsonResponse(400, "Invalid Hugging Face endpoint configuration.", endpointValidation.details);
  }

  const form = await request.formData();
  const file = form.get("image");
  if (!(file instanceof File)) {
    return jsonResponse(400, "No image uploaded.", "Send a file under the `image` form field.");
  }

  const requestMode = process.env.HF_REQUEST_MODE?.trim().toLowerCase() || "multipart";
  const method = process.env.HF_HTTP_METHOD?.trim().toUpperCase() || "POST";
  const imageFieldName = process.env.HF_IMAGE_FIELD_NAME?.trim() || "image";
  if (!["multipart", "raw", "json-base64"].includes(requestMode)) {
    return jsonResponse(
      500,
      "Invalid frontend environment configuration.",
      "HF_REQUEST_MODE must be one of: multipart, raw, json-base64."
    );
  }
  let extraFields: JsonRecord;
  try {
    extraFields = parseExtraFields();
  } catch (error) {
    return jsonResponse(
      500,
      "Invalid frontend environment configuration.",
      error instanceof Error ? error.message : "Failed to parse HF_EXTRA_FIELDS_JSON."
    );
  }
  const bytes = Buffer.from(await file.arrayBuffer());

  const headers = new Headers();
  const token = process.env.HF_API_TOKEN?.trim();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  headers.set("Accept", "image/*,application/json");

  let body: BodyInit;

  if (requestMode === "raw") {
    headers.set("Content-Type", file.type || "application/octet-stream");
    body = bytes;
  } else if (requestMode === "json-base64") {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify({
      ...extraFields,
      inputs: bytes.toString("base64")
    });
  } else {
    const upstreamForm = new FormData();
    upstreamForm.append(
      imageFieldName,
      new Blob([bytes], { type: file.type || "application/octet-stream" }),
      file.name
    );

    for (const [key, value] of Object.entries(extraFields)) {
      upstreamForm.append(key, typeof value === "string" ? value : JSON.stringify(value));
    }

    body = upstreamForm;
  }

  let upstream: Response;
  try {
    upstream = await fetch(endpoint, {
      method,
      headers,
      body,
      cache: "no-store"
    });
  } catch (error) {
    return jsonResponse(
      502,
      "Could not reach the Hugging Face endpoint.",
      error instanceof Error ? error.message : "Network request failed."
    );
  }

  if (!upstream.ok) {
    const details = await upstream.text().catch(() => "");
    return jsonResponse(
      upstream.status,
      "Hugging Face request failed.",
      details || `Endpoint responded with ${upstream.status}.`
    );
  }

  const contentType = upstream.headers.get("content-type") || "";

  if (contentType.startsWith("image/")) {
    const resultBytes = await upstream.arrayBuffer();
    return new Response(resultBytes, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": 'inline; filename="denoised-output.png"',
        "Cache-Control": "no-store"
      }
    });
  }

  if (contentType.includes("application/json")) {
    const payload = await upstream.json().catch(() => null);
    const decoded = extractImageFromJson(payload);
    if (!decoded) {
      return jsonResponse(
        502,
        "Unsupported Hugging Face JSON response.",
        "The endpoint returned JSON, but no image payload could be extracted."
      );
    }

    return new Response(new Blob([decoded]), {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "Content-Disposition": 'inline; filename="denoised-output.png"',
        "Cache-Control": "no-store"
      }
    });
  }

  const fallbackText = await upstream.text().catch(() => "");
  return jsonResponse(
    502,
    "Unsupported Hugging Face response type.",
    fallbackText || `Received content-type: ${contentType}`
  );
}
