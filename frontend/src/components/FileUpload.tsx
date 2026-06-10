"use client";

import { useRef, useState } from "react";
import { Upload, CheckCircle, AlertCircle, File } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uploadDocument, DuplicateDocumentError, type DocumentInfo, type UploadResponse } from "@/lib/api";

const MAX_BYTES = 50 * 1024 * 1024;

interface Props {
  onUploadSuccess: (doc: DocumentInfo) => void;
  onUploadError?: (err: unknown) => void;
}

type State =
  | { phase: "idle" }
  | { phase: "selected"; file: File }
  | { phase: "uploading"; file: File; pct: number }
  | { phase: "success"; result: UploadResponse }
  | { phase: "error"; message: string };

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileUpload({ onUploadSuccess, onUploadError }: Props) {
  const [state, setState] = useState<State>({ phase: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function pickFile(file: File) {
    if (!file.name.match(/\.(pdf|txt)$/i)) {
      setState({ phase: "error", message: "Only PDF and TXT files are supported." });
      return;
    }
    if (file.size > MAX_BYTES) {
      setState({ phase: "error", message: "File exceeds 50 MB limit." });
      return;
    }
    setState({ phase: "selected", file });
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) pickFile(file);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) pickFile(file);
  }

  async function upload() {
    if (state.phase !== "selected") return;
    const { file } = state;
    setState({ phase: "uploading", file, pct: 0 });
    try {
      const result = await uploadDocument(file, (pct) =>
        setState({ phase: "uploading", file, pct })
      );
      setState({ phase: "success", result });
      onUploadSuccess({
        doc_id: result.doc_id,
        filename: result.filename,
        chunk_count: result.chunk_count,
        page_count: result.page_count,
        uploaded_at: new Date().toISOString(),
        status: "ready",
        file_size_bytes: null,
        content_hash: null,
      });
    } catch (err) {
      if (err instanceof DuplicateDocumentError && onUploadError) {
        setState({ phase: "idle" });
        if (inputRef.current) inputRef.current.value = "";
        onUploadError(err);
      } else {
        setState({ phase: "error", message: err instanceof Error ? err.message : "Upload failed" });
        if (onUploadError) onUploadError(err);
      }
    }
  }

  function reset() {
    setState({ phase: "idle" });
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onClick={() => state.phase === "idle" && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          "relative flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 transition-colors",
          state.phase === "idle" || state.phase === "selected"
            ? "cursor-pointer"
            : "cursor-default",
          dragging
            ? "border-[var(--accent-blue)] bg-[var(--accent-blue)]/5"
            : "border-[var(--border-subtle)] hover:border-[var(--border-muted)]",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt"
          className="sr-only"
          onChange={onInputChange}
        />

        {state.phase === "idle" && (
          <>
            <Upload className="h-8 w-8 text-[var(--text-muted)]" />
            <p className="text-sm text-[var(--text-muted)]">
              Drag & drop or <span className="text-[var(--accent-blue)] underline-offset-2 hover:underline">browse</span>
            </p>
            <p className="text-xs text-[var(--text-subtle)]">PDF or TXT · Max 50 MB</p>
          </>
        )}

        {(state.phase === "selected") && (
          <>
            <File className="h-7 w-7 text-[var(--accent-blue)]" />
            <p className="max-w-full truncate text-sm font-medium text-[var(--text-primary)]">
              {state.file.name}
            </p>
            <p className="text-xs text-[var(--text-muted)]">{formatBytes(state.file.size)}</p>
          </>
        )}

        {state.phase === "uploading" && (
          <>
            <File className="h-7 w-7 text-[var(--accent-blue)]" />
            <p className="max-w-full truncate text-sm font-medium text-[var(--text-primary)]">
              {state.file.name}
            </p>
            <div className="mt-1 w-full max-w-[200px]">
              <div className="h-1 w-full overflow-hidden rounded-full bg-[var(--border-subtle)]">
                <div
                  className="h-full rounded-full bg-[var(--accent-blue)] transition-all duration-200"
                  style={{ width: `${state.pct}%` }}
                />
              </div>
              <p className="mt-1 text-center text-xs text-[var(--text-muted)]">{state.pct}%</p>
            </div>
          </>
        )}

        {state.phase === "success" && (
          <>
            <CheckCircle className="h-7 w-7 text-emerald-500" />
            <p className="max-w-full truncate text-sm font-medium text-[var(--text-primary)]">
              {state.result.filename}
            </p>
            <p className="text-xs text-[var(--text-muted)]">
              {state.result.chunk_count} chunks · {state.result.page_count} pages
            </p>
          </>
        )}

        {state.phase === "error" && (
          <>
            <AlertCircle className="h-7 w-7 text-red-500" />
            <p className="text-center text-sm text-red-400">{state.message}</p>
          </>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        {state.phase === "selected" && (
          <>
            <Button
              onClick={upload}
              size="sm"
              className="flex-1 bg-[var(--accent-blue)] text-white hover:bg-[var(--accent-blue-hover)]"
            >
              Upload
            </Button>
            <Button onClick={reset} size="sm" variant="outline" className="border-[var(--border-subtle)] text-[var(--text-muted)]">
              Cancel
            </Button>
          </>
        )}
        {(state.phase === "success" || state.phase === "error") && (
          <Button
            onClick={reset}
            size="sm"
            variant="outline"
            className="w-full border-[var(--border-subtle)] text-[var(--text-muted)]"
          >
            Upload another
          </Button>
        )}
      </div>
    </div>
  );
}
