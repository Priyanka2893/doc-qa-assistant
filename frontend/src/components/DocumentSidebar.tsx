"use client";

import { useState } from "react";
import { Trash2, FileText, Globe, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { type DocumentInfo } from "@/lib/api";

interface Props {
  documents: DocumentInfo[];
  activeDocId: string | null;
  isGlobal: boolean;
  onSelect: (docId: string) => void;
  onSelectGlobal: () => void;
  onDelete: (docId: string) => void;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function formatFileSize(bytes: number | null): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function StatusBadge({ status }: { status: string }) {
  if (status === "processing") {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-yellow-400">
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
        processing
      </span>
    );
  }
  if (status === "error") {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-red-400">
        <AlertCircle className="h-2.5 w-2.5" />
        error
      </span>
    );
  }
  return (
    <span className="flex items-center gap-0.5 text-[10px] text-emerald-400">
      <CheckCircle2 className="h-2.5 w-2.5" />
      ready
    </span>
  );
}

export default function DocumentSidebar({
  documents,
  activeDocId,
  isGlobal,
  onSelect,
  onSelectGlobal,
  onDelete,
}: Props) {
  const [confirmId, setConfirmId] = useState<string | null>(null);

  function handleDeleteClick(e: React.MouseEvent, docId: string) {
    e.stopPropagation();
    setConfirmId(docId);
  }

  function confirmDelete(docId: string) {
    onDelete(docId);
    setConfirmId(null);
  }

  return (
    <div className="flex flex-col gap-1 px-2 py-1">
      {/* Global search button */}
      {documents.length > 0 && (
        <div
          role="button"
          tabIndex={0}
          onClick={onSelectGlobal}
          onKeyDown={(e) => e.key === "Enter" && onSelectGlobal()}
          className={[
            "flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-2 transition-colors",
            isGlobal
              ? "border border-[var(--accent-blue)]/50 bg-[var(--accent-blue)]/10"
              : "border border-transparent hover:bg-[var(--surface-hover)]",
          ].join(" ")}
        >
          <Globe
            className={[
              "h-4 w-4 shrink-0",
              isGlobal ? "text-[var(--accent-blue)]" : "text-[var(--text-muted)]",
            ].join(" ")}
          />
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-[var(--text-primary)]">Global Search</p>
            <p className="text-[10px] text-[var(--text-subtle)]">
              Search across all {documents.length} doc{documents.length !== 1 ? "s" : ""}
            </p>
          </div>
        </div>
      )}

      {documents.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
          <FileText className="h-8 w-8 text-[var(--text-subtle)]" />
          <p className="text-sm text-[var(--text-muted)]">No documents yet.</p>
          <p className="text-xs text-[var(--text-subtle)]">Upload one to get started.</p>
        </div>
      ) : (
        <ul className="space-y-1">
          {documents.map((doc) => {
            const isActive = !isGlobal && doc.doc_id === activeDocId;
            const isConfirming = confirmId === doc.doc_id;
            const sizeLabel = formatFileSize(doc.file_size_bytes);

            return (
              <li key={doc.doc_id}>
                {isConfirming ? (
                  <div className="rounded-md border border-red-500/40 bg-red-500/10 p-2.5">
                    <p className="mb-2 text-xs text-red-400">Delete this document?</p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => confirmDelete(doc.doc_id)}
                        className="flex-1 rounded bg-red-500 px-2 py-1 text-xs font-medium text-white transition-colors hover:bg-red-600"
                      >
                        Delete
                      </button>
                      <button
                        onClick={() => setConfirmId(null)}
                        className="flex-1 rounded border border-[var(--border-subtle)] px-2 py-1 text-xs text-[var(--text-muted)] transition-colors hover:border-[var(--border-muted)]"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => onSelect(doc.doc_id)}
                    onKeyDown={(e) => e.key === "Enter" && onSelect(doc.doc_id)}
                    className={[
                      "group flex w-full cursor-pointer items-start gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors",
                      isActive
                        ? "border border-[var(--accent-blue)]/50 bg-[var(--accent-blue)]/10"
                        : "border border-transparent hover:bg-[var(--surface-hover)]",
                    ].join(" ")}
                  >
                    <FileText
                      className={[
                        "mt-0.5 h-4 w-4 shrink-0",
                        isActive ? "text-[var(--accent-blue)]" : "text-[var(--text-muted)]",
                      ].join(" ")}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-[var(--text-primary)]">
                        {doc.filename}
                      </p>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5">
                        <Badge
                          variant="secondary"
                          className="h-4 rounded px-1 py-0 text-[10px] leading-none"
                        >
                          {doc.chunk_count} chunks
                        </Badge>
                        {sizeLabel && (
                          <span className="text-[10px] text-[var(--text-subtle)]">{sizeLabel}</span>
                        )}
                        <span className="text-[10px] text-[var(--text-subtle)]">
                          {relativeTime(doc.uploaded_at)}
                        </span>
                        <StatusBadge status={doc.status} />
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteClick(e, doc.doc_id)}
                      className="shrink-0 rounded p-0.5 opacity-0 transition-opacity group-hover:opacity-100 hover:text-red-400"
                      aria-label="Delete document"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-[var(--text-muted)]" />
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
