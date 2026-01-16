'use client';

import { useEffect, useRef, useState } from "react";
import hljs from "highlight.js/lib/core";
import csharp from "highlight.js/lib/languages/csharp";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import "highlight.js/styles/github-dark.css";

hljs.registerLanguage("csharp", csharp);
hljs.registerLanguage("cs", csharp);
hljs.registerLanguage("java", java);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("js", javascript);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("ts", typescript);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("html", xml);

interface ChunkMeta {
  [key: string]: unknown;
  file_path?: string;
  start_line?: number;
  end_line?: number;
  language?: string;
  text?: string;
  chunk_type?: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChunkMeta[];
}

interface Props {
  repoId: string;
  question: string;
  messages: Message[];
  loading: boolean;
  disabledReason?: string | null;
  onQuestionChange: (value: string) => void;
  onAsk: (prompt: string) => void;
}

export function ChatPanel({
  repoId,
  question,
  messages,
  loading,
  disabledReason,
  onQuestionChange,
  onAsk,
}: Props) {
  const [error, setError] = useState<string | null>(null);
  const disabled = !!disabledReason || !repoId;
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  function splitSources(content: string): { text: string; sources: string | null } {
    const marker = "```sources";
    const start = content.indexOf(marker);
    if (start === -1) return { text: content, sources: null };
    const end = content.indexOf("```", start + marker.length);
    if (end === -1) return { text: content, sources: null };

    const text = content.slice(0, start).trim();
    const sources = content
      .slice(start + marker.length, end)
      .replace(/^\s*\n/, "")
      .trim();
    return { text, sources: sources || null };
  }

  useEffect(() => {
    const blocks = document.querySelectorAll<HTMLElement>(
      ".chat-messages pre code"
    );
    blocks.forEach((block) => {
      hljs.highlightElement(block);
    });
  }, [messages]);

  useEffect(() => {
    if (!messagesEndRef.current) return;
    messagesEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function inferLanguage(filePath?: string, language?: string | null): string | null {
    if (language && language.toLowerCase() != "unknown") return language;
    if (!filePath) return null;
    const ext = filePath.split(".").pop()?.toLowerCase();
    if (!ext) return null;
    const map: Record<string, string> = {
      ts: "typescript",
      tsx: "tsx",
      js: "javascript",
      jsx: "jsx",
      py: "python",
      rs: "rust",
      go: "go",
      java: "java",
      cs: "csharp",
      cshtml: "html",
      c: "c",
      cpp: "cpp",
      h: "c",
      md: "markdown",
      yml: "yaml",
      yaml: "yaml",
      json: "json",
      css: "css",
      scss: "scss",
      html: "html",
    };
    return map[ext] ?? ext;
  }

  type Segment = { type: "text" | "code"; language?: string | null; content: string };

  function parseCodeBlocks(text: string): Segment[] {
    const segments: Segment[] = [];
    const regex = /```(\w+)?\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        segments.push({ type: "text", content: text.slice(lastIndex, match.index).trim() });
      }
      segments.push({
        type: "code",
        language: match[1]?.toLowerCase() ?? null,
        content: match[2].trimEnd(),
      });
      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      segments.push({ type: "text", content: text.slice(lastIndex).trim() });
    }

    if (segments.length === 0) {
      segments.push({ type: "text", content: text });
    }

    return segments.filter((segment) => segment.content.length > 0);
  }

  function submitPrompt(event?: React.FormEvent<HTMLFormElement>) {
    if (event) event.preventDefault();
    if (disabled || loading) return;
    const trimmed = question.trim();
    if (!trimmed) {
      setError("Enter a question");
      return;
    }
    setError(null);
    onAsk(trimmed);
    onQuestionChange("");
  }

  return (
    <section className={`chat-window ${disabled ? "is-blocked" : ""}`}>
      {disabled && (
        <div className="chat-overlay">
          <div className="chat-overlay__content">
            {disabledReason?.toLowerCase().includes("index") && <span className="overlay-spinner" />}
            <p>{disabledReason || "Select a repository to start chatting."}</p>
          </div>
        </div>
      )}
      <div className="chat-messages" aria-hidden={disabled}>
        {messages.length === 0 ? (
          <div className="message ai">Ask about the selected repository to see contextual answers.</div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`message ${message.role === 'user' ? 'user' : 'ai'}`}>
              {message.role === "assistant" ? (() => {
                const { text, sources } = splitSources(message.content);
                const segments = parseCodeBlocks(text);
                const structuredSources = (message.sources as ChunkMeta[] | undefined) || [];
                const codeSources = structuredSources.filter(
                  (chunk) => String(chunk.chunk_type || "code").toLowerCase() === "code"
                );
                return (
                  <>
                    {segments.map((segment, index) =>
                      segment.type === "code" ? (
                        <pre
                          key={`code-${message.id}-${index}`}
                          className={`code-block ${segment.language ? `code-block--${segment.language}` : ""}`.trim()}
                        >
                          <code className={`mono ${segment.language ? `language-${segment.language}` : ""}`}>
                            {segment.content}
                          </code>
                        </pre>
                      ) : (
                        <div key={`text-${message.id}-${index}`} className="message-text">
                          {segment.content}
                        </div>
                      )
                    )}
                    {codeSources.length > 0
                      ? codeSources.map((chunk, index) => {
                          const filePath = String(chunk.file_path || "Source snippet");
                          const language = inferLanguage(filePath, typeof chunk.language === "string" ? chunk.language : null);
                          const lineInfo =
                            typeof chunk.start_line === "number" && typeof chunk.end_line === "number"
                              ? `L${chunk.start_line}-${chunk.end_line}`
                              : null;
                          const codeText = typeof chunk.text === "string" ? chunk.text : "";
                          return (
                            <div key={`source-${message.id}-${index}`} className="source-entry" aria-label="Sources">
                              <div className="source-entry__header">
                                <span className="source-entry__file mono" title={filePath}>
                                  {filePath}
                                  {lineInfo ? ` (${lineInfo})` : ""}
                                </span>
                                {language && <span className="source-entry__lang">{language}</span>}
                              </div>
                              <pre className={`code-block ${language ? `code-block--${language}` : ""}`.trim()}>
                                <code className={`mono ${language ? `language-${language}` : ""}`}>
                                  {codeText || "[source text unavailable]"}
                                </code>
                              </pre>
                            </div>
                          );
                        })
                      : sources && (
                          <pre className="sources-block" aria-label="Sources">
                            <code className="mono">{sources}</code>
                          </pre>
                        )}
                  </>
                );
              })() : (
                <div className="message-text">{message.content}</div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-form" onSubmit={submitPrompt} aria-disabled={disabled}>
        <textarea
          rows={3}
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="Ask anything about this repo…"
          disabled={disabled}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !disabled) {
              event.preventDefault();
              submitPrompt();
            }
          }}
        />
        {error && <div className="error-text">{error}</div>}
        <button className="primary" type="submit" disabled={disabled || loading}>
          {loading ? "Generating…" : "Ask"}
        </button>
      </form>
    </section>
  );
}
