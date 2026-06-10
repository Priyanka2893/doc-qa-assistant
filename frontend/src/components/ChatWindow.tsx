"use client";

import { useEffect, useRef, useState } from "react";
import { Send, MessageSquare, Loader2, Globe } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import MessageBubble from "@/components/MessageBubble";
import { type ChunkSource, type GlobalChunkSource } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: ChunkSource[] | GlobalChunkSource[];
  isLoading?: boolean;
  isGlobal?: boolean;
}

interface Props {
  messages: Message[];
  onSend: (text: string) => void;
  isLoading: boolean;
  disabled: boolean;
  isGlobal?: boolean;
  documentCount?: number;
}

export default function ChatWindow({
  messages,
  onSend,
  isLoading,
  disabled,
  isGlobal = false,
  documentCount = 0,
}: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function submit() {
    const text = input.trim();
    if (!text || isLoading || disabled) return;
    setInput("");
    onSend(text);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function onInputChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  const placeholder = disabled
    ? "Select a document first…"
    : isGlobal
    ? "Ask anything across all documents… (Enter to send)"
    : "Ask a question… (Enter to send)";

  return (
    <div className="flex h-full flex-col">
      {/* Global search banner */}
      {isGlobal && (
        <div className="flex items-center gap-2 border-b border-[var(--border-subtle)] bg-[var(--accent-blue)]/5 px-4 py-2">
          <Globe className="h-3.5 w-3.5 shrink-0 text-[var(--accent-blue)]" />
          <p className="text-xs text-[var(--accent-blue)]">
            Searching across all {documentCount} document{documentCount !== 1 ? "s" : ""}
          </p>
        </div>
      )}

      {/* Message list */}
      <ScrollArea className="flex-1 px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full min-h-[300px] flex-col items-center justify-center gap-3 text-center">
            <div className="rounded-full border border-[var(--border-subtle)] p-4">
              {isGlobal ? (
                <Globe className="h-7 w-7 text-[var(--accent-blue)]" />
              ) : (
                <MessageSquare className="h-7 w-7 text-[var(--text-muted)]" />
              )}
            </div>
            <p className="text-sm text-[var(--text-muted)]">
              {disabled
                ? "Select a document to start chatting."
                : isGlobal
                ? `Ask anything — searching across all ${documentCount} document${documentCount !== 1 ? "s" : ""}.`
                : "Ask anything about your document."}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                role={msg.role}
                content={msg.content}
                timestamp={msg.timestamp}
                sources={msg.sources}
                isLoading={msg.isLoading}
                isGlobal={msg.isGlobal}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </ScrollArea>

      {/* Input bar */}
      <div className="border-t border-[var(--border-subtle)] px-4 py-3">
        <div
          className={[
            "chat-input-wrapper flex items-end gap-2 rounded-xl border bg-[var(--surface-card)] px-3 py-2 transition-all",
            disabled ? "opacity-50" : "",
          ].join(" ")}
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={onInputChange}
            onKeyDown={onKeyDown}
            disabled={disabled || isLoading}
            placeholder={placeholder}
            className="flex-1 resize-none bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-subtle)] focus:outline-none disabled:cursor-not-allowed"
          />
          <button
            onClick={submit}
            disabled={!input.trim() || isLoading || disabled}
            className="mb-0.5 shrink-0 rounded-lg p-1.5 text-[var(--accent-blue)] transition-colors hover:bg-[var(--accent-blue)]/10 disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Send"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-[var(--text-subtle)]">
          Shift + Enter for new line
        </p>
      </div>
    </div>
  );
}
