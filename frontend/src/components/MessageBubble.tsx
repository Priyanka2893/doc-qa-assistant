"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, FileText, Globe } from "lucide-react";
import { type ChunkSource, type GlobalChunkSource } from "@/lib/api";

function isGlobalSource(s: ChunkSource | GlobalChunkSource): s is GlobalChunkSource {
  return "filename" in s;
}

interface Props {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: ChunkSource[] | GlobalChunkSource[];
  isLoading?: boolean;
  isGlobal?: boolean;
}

function formatTime(d: Date) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-2">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-[var(--text-muted)] animate-bounce"
          style={{ animationDelay: `${i * 150}ms`, animationDuration: "1s" }}
        />
      ))}
    </div>
  );
}

function SkeletonLine({ width }: { width: string }) {
  return (
    <div
      className="h-3 animate-pulse rounded bg-[var(--border-subtle)]"
      style={{ width }}
    />
  );
}

export default function MessageBubble({ role, content, timestamp, sources, isLoading, isGlobal }: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isUser = role === "user";

  if (isLoading) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[75%] rounded-2xl rounded-tl-sm bg-[var(--surface-card)] px-4 py-3">
          {isUser ? (
            <TypingIndicator />
          ) : (
            <div className="space-y-2">
              <SkeletonLine width="80%" />
              <SkeletonLine width="60%" />
              <SkeletonLine width="70%" />
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={["flex flex-col gap-1", isUser ? "items-end" : "items-start"].join(" ")}>
      <div
        className={[
          "max-w-[75%] rounded-2xl px-4 py-2.5",
          isUser
            ? "rounded-tr-sm bg-[var(--accent-blue)] text-white"
            : "rounded-tl-sm bg-[var(--surface-card)] text-[var(--text-primary)]",
        ].join(" ")}
      >
        <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{content}</p>

        {!isUser && sources && sources.length > 0 && (
          <div className="mt-3 border-t border-white/10 pt-2">
            <button
              onClick={() => setSourcesOpen((v) => !v)}
              className="flex items-center gap-1 text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
            >
              {sourcesOpen ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
              {isGlobal ? <Globe className="h-3 w-3" /> : <FileText className="h-3 w-3" />}
              {sources.length} source{sources.length !== 1 ? "s" : ""}
              {isGlobal && " (all docs)"}
            </button>

            {sourcesOpen && (
              <ul className="mt-2 space-y-2">
                {sources.map((s, i) => (
                  <li
                    key={`${s.chunk_index}-${i}`}
                    className="rounded-md border border-[var(--border-subtle)] bg-[var(--surface-base)] p-2"
                  >
                    {isGlobalSource(s) && (
                      <div className="mb-1.5 flex items-center gap-1">
                        <FileText className="h-3 w-3 text-[var(--accent-blue)]" />
                        <span className="text-[10px] font-medium text-[var(--accent-blue)]">
                          {s.filename}
                        </span>
                      </div>
                    )}
                    <p className="line-clamp-3 text-xs text-[var(--text-muted)]">
                      {s.text_excerpt}
                    </p>
                    <div className="mt-1.5 flex items-center gap-2">
                      <span className="text-[10px] text-[var(--accent-blue)]">
                        {Math.round(s.score * 100)}% match
                      </span>
                      {s.page_number != null && (
                        <span className="text-[10px] text-[var(--text-subtle)]">
                          p. {s.page_number}
                        </span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      <span className="px-1 text-[10px] text-[var(--text-subtle)]">{formatTime(timestamp)}</span>
    </div>
  );
}
