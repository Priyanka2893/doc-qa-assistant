export interface ChunkSource {
  chunk_index: number;
  text_excerpt: string;
  score: number;
  page_number: number | null;
}

export interface GlobalChunkSource extends ChunkSource {
  filename: string;
  doc_id: string;
}

export interface UploadResponse {
  doc_id: string;
  filename: string;
  chunk_count: number;
  page_count: number;
  status: string;
  ingestion_time_ms: number;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  chunk_count: number;
  page_count: number;
  uploaded_at: string;
  status: string;
  file_size_bytes: number | null;
  content_hash: string | null;
}

export interface AskRequest {
  question: string;
  document_id: string;
  top_k?: number;
  session_id?: string;
}

export interface AskResponse {
  answer: string;
  sources: ChunkSource[];
  model: string;
  tokens_used: number;
  doc_id: string;
  cache_hit: boolean;
  session_id?: string;
}

export interface GlobalAskRequest {
  question: string;
  top_k?: number;
}

export interface GlobalAskResponse {
  answer: string;
  sources: GlobalChunkSource[];
  model: string;
  tokens_used: number;
}

export interface HealthResponse {
  status: string;
  qdrant: string;
  embedding_model: string;
  version: string;
}

export class DuplicateDocumentError extends Error {
  existing_doc_id: string;
  existing_filename: string;

  constructor(existing_doc_id: string, existing_filename: string) {
    super("Document already exists");
    this.name = "DuplicateDocumentError";
    this.existing_doc_id = existing_doc_id;
    this.existing_filename = existing_filename;
  }
}

const BASE = "/api/backend";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.open("POST", `${BASE}/documents/upload`);

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch {
          reject(new Error("Invalid JSON response"));
        }
      } else if (xhr.status === 409) {
        try {
          const body = JSON.parse(xhr.responseText);
          reject(new DuplicateDocumentError(body.existing_doc_id ?? "", body.filename ?? ""));
        } catch {
          reject(new DuplicateDocumentError("", ""));
        }
      } else {
        let message = `HTTP ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          message = body.detail ?? body.message ?? message;
        } catch {
          // ignore
        }
        reject(new Error(message));
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE}/documents`);
  return handleResponse<DocumentInfo[]>(res);
}

export async function getDocument(docId: string): Promise<DocumentInfo> {
  const res = await fetch(`${BASE}/documents/${docId}`);
  return handleResponse<DocumentInfo>(res);
}

export async function askQuestion(request: AskRequest): Promise<AskResponse> {
  const res = await fetch(`${BASE}/qa/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse<AskResponse>(res);
}

export async function askGlobal(request: GlobalAskRequest): Promise<GlobalAskResponse> {
  const res = await fetch(`${BASE}/qa/ask-global`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse<GlobalAskResponse>(res);
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      // ignore
    }
    throw new Error(message);
  }
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export function askQuestionStream(
  request: AskRequest,
  onChunk: (text: string) => void,
  onSources: (sources: ChunkSource[]) => void,
  onDone: (sessionId?: string) => void,
  onError?: (error: Error) => void
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/qa/ask-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
      }

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6)) as {
              type: string;
              text?: string;
              sources?: ChunkSource[];
              tokens_used?: number;
              session_id?: string;
              detail?: string;
            };
            if (event.type === "chunk" && event.text) {
              onChunk(event.text);
            } else if (event.type === "sources" && event.sources) {
              onSources(event.sources);
            } else if (event.type === "done") {
              onDone(event.session_id);
            } else if (event.type === "error") {
              throw new Error(event.detail ?? "Stream error");
            }
          } catch (e) {
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  })();

  return () => controller.abort();
}
