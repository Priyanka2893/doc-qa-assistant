"use client";

import { useCallback, useEffect, useState } from "react";
import { nanoid } from "nanoid";
import { toast } from "sonner";
import { FileText, Menu, X, Globe } from "lucide-react";
import FileUpload from "@/components/FileUpload";
import DocumentSidebar from "@/components/DocumentSidebar";
import ChatWindow, { type Message } from "@/components/ChatWindow";
import {
  listDocuments,
  deleteDocument,
  askQuestionStream,
  askGlobal,
  healthCheck,
  DuplicateDocumentError,
  type DocumentInfo,
  type ChunkSource,
} from "@/lib/api";

export default function Home() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [isGlobal, setIsGlobal] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const activeDoc = documents.find((d) => d.doc_id === activeDocId) ?? null;

  /* ── Load documents + health on mount ── */
  useEffect(() => {
    listDocuments()
      .then(setDocuments)
      .catch(() => toast.error("Failed to load documents"));

    healthCheck()
      .then((h) => setHealthy(h.status === "ok"))
      .catch(() => setHealthy(false));
  }, []);

  /* ── Select a specific document → clear global mode ── */
  function selectDoc(docId: string) {
    setActiveDocId(docId);
    setIsGlobal(false);
    setMessages([]);
    setSidebarOpen(false);
  }

  /* ── Activate global search mode ── */
  function selectGlobal() {
    setActiveDocId(null);
    setIsGlobal(true);
    setMessages([]);
    setSidebarOpen(false);
  }

  /* ── Delete document ── */
  async function handleDelete(docId: string) {
    try {
      await deleteDocument(docId);
      setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
      if (activeDocId === docId) {
        setActiveDocId(null);
        setIsGlobal(false);
        setMessages([]);
      }
      toast.success("Document deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed");
    }
  }

  /* ── Upload success ── */
  function handleUploadSuccess(doc: DocumentInfo) {
    setDocuments((prev) => {
      const exists = prev.some((d) => d.doc_id === doc.doc_id);
      return exists ? prev : [doc, ...prev];
    });
    setActiveDocId(doc.doc_id);
    setIsGlobal(false);
    setMessages([]);
    toast.success(`"${doc.filename}" ingested successfully`);
  }

  /* ── Upload error (including duplicate) ── */
  function handleUploadError(err: unknown) {
    if (err instanceof DuplicateDocumentError) {
      toast.error(`"${err.existing_filename}" was already uploaded`, {
        description: "This file already exists in your library.",
        action: {
          label: "Select it",
          onClick: () => selectDoc(err.existing_doc_id),
        },
      });
    } else {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    }
  }

  /* ── Send question ── */
  const handleSend = useCallback(
    (text: string) => {
      if (isAsking) return;
      if (!activeDocId && !isGlobal) return;

      const userMsg: Message = {
        id: nanoid(),
        role: "user",
        content: text,
        timestamp: new Date(),
      };
      const loadingId = nanoid();
      const loadingMsg: Message = {
        id: loadingId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isLoading: true,
        isGlobal,
      };

      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setIsAsking(true);

      if (isGlobal) {
        askGlobal({ question: text, top_k: 10 })
          .then((res) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === loadingId
                  ? { ...m, content: res.answer, sources: res.sources, isLoading: false }
                  : m
              )
            );
          })
          .catch((err) => {
            setMessages((prev) => prev.filter((m) => m.id !== loadingId));
            toast.error(err instanceof Error ? err.message : "Failed to get answer");
          })
          .finally(() => setIsAsking(false));
      } else if (activeDocId) {
        let accumulated = "";
        askQuestionStream(
          { question: text, document_id: activeDocId },
          (chunk) => {
            accumulated += chunk;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === loadingId ? { ...m, content: accumulated, isLoading: false } : m
              )
            );
          },
          (sources: ChunkSource[]) => {
            setMessages((prev) =>
              prev.map((m) => (m.id === loadingId ? { ...m, sources } : m))
            );
          },
          () => setIsAsking(false),
          (err) => {
            setMessages((prev) => prev.filter((m) => m.id !== loadingId));
            toast.error(err.message);
            setIsAsking(false);
          }
        );
      }
    },
    [activeDocId, isGlobal, isAsking]
  );

  const chatDisabled = !activeDocId && !isGlobal;

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Mobile sidebar overlay ── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Left sidebar ── */}
      <aside
        className={[
          "fixed inset-y-0 left-0 z-30 flex w-[280px] flex-col border-r border-[var(--border-subtle)] bg-[var(--surface-card)] transition-transform duration-200 lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        {/* Sidebar header */}
        <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-4 py-3">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-[var(--accent-blue)]" />
            <span className="text-sm font-semibold text-[var(--text-primary)]">Documents</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Upload area */}
        <div className="border-b border-[var(--border-subtle)] p-3">
          <FileUpload
            onUploadSuccess={handleUploadSuccess}
            onUploadError={handleUploadError}
          />
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto py-2">
          <DocumentSidebar
            documents={documents}
            activeDocId={activeDocId}
            isGlobal={isGlobal}
            onSelect={selectDoc}
            onSelectGlobal={selectGlobal}
            onDelete={handleDelete}
          />
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top header */}
        <header className="flex shrink-0 items-center gap-3 border-b border-[var(--border-subtle)] bg-[var(--surface-card)] px-4 py-3">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden text-[var(--text-muted)] hover:text-[var(--text-primary)]"
            aria-label="Open sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded bg-[var(--accent-blue)]">
              <FileText className="h-3.5 w-3.5 text-white" />
            </div>
            <span className="text-sm font-bold tracking-tight text-[var(--text-primary)]">
              Doc Q&amp;A
            </span>
          </div>

          {/* Active context label */}
          {isGlobal ? (
            <>
              <span className="text-[var(--border-muted)]">/</span>
              <span className="flex items-center gap-1 text-sm text-[var(--accent-blue)]">
                <Globe className="h-3.5 w-3.5" />
                Global Search
              </span>
            </>
          ) : activeDoc ? (
            <>
              <span className="text-[var(--border-muted)]">/</span>
              <span className="max-w-[200px] truncate text-sm text-[var(--text-muted)]">
                {activeDoc.filename}
              </span>
            </>
          ) : null}

          {/* Spacer */}
          <div className="flex-1" />

          {/* Health dot */}
          <div className="flex items-center gap-1.5">
            <div
              className={[
                "h-2 w-2 rounded-full",
                healthy === null
                  ? "bg-[var(--text-subtle)]"
                  : healthy
                  ? "bg-emerald-500"
                  : "bg-red-500",
              ].join(" ")}
            />
            <span className="hidden text-xs text-[var(--text-muted)] sm:block">
              {healthy === null ? "connecting…" : healthy ? "online" : "offline"}
            </span>
          </div>
        </header>

        {/* Chat area */}
        <main className="flex-1 overflow-hidden">
          <ChatWindow
            messages={messages}
            onSend={handleSend}
            isLoading={isAsking}
            disabled={chatDisabled}
            isGlobal={isGlobal}
            documentCount={documents.length}
          />
        </main>
      </div>
    </div>
  );
}
