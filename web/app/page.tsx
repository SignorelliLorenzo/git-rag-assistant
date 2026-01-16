'use client';

import { useEffect, useMemo, useState } from "react";
import { LoginPanel } from "@/components/LoginPanel";
import { ChatPanel } from "@/components/ChatPanel";
import { RepoToolbar } from "@/components/RepoToolbar";
import {
  askQuestion,
  getSession,
  indexRepo,
  listRepos,
  deleteRepo,
} from "@/lib/api";
import type {
  AskResponse,
  LoginResponse,
  RepoInfo,
} from "@/lib/types";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<Record<string, unknown>>;
}

export default function HomePage() {
  const [session, setSession] = useState<LoginResponse | null>(null);
  const [repos, setRepos] = useState<RepoInfo[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<string>("");
  const [status, setStatus] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [browseOpen, setBrowseOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredRepos, setFilteredRepos] = useState<RepoInfo[]>([]);
  const [deleteBusy, setDeleteBusy] = useState<string | null>(null);

  useEffect(() => {
    const token = window.localStorage.getItem("session_token");
    if (!token) return;
    getSession(token)
      .then((session) => setSession(session))
      .catch(() => {
        window.localStorage.removeItem("session_token");
      });
  }, []);

  const localRepos = useMemo(() => repos.filter((repo) => Boolean(repo.repo_path)), [repos]);
  const remoteRepos = useMemo(() => repos.filter((repo) => !repo.repo_path), [repos]);

  useEffect(() => {
    if (!session) {
      setRepos([]);
      return;
    }
    refreshRepos();
  }, [session]);

  useEffect(() => {
    if (!browseOpen) {
      if (filteredRepos.length) setFilteredRepos([]);
      if (searchTerm) setSearchTerm("");
      return;
    }
    const normalized = searchTerm.trim().toLowerCase();
    if (!normalized) {
      setFilteredRepos(remoteRepos);
      return;
    }
    setFilteredRepos(
      remoteRepos.filter((repo) => extractRepoName(repo).toLowerCase().includes(normalized))
    );
  }, [browseOpen, remoteRepos, searchTerm, filteredRepos.length]);

  async function refreshRepos() {
    if (!session) return;
    try {
      const { repos: apiRepos } = await listRepos(session.session_token);
      setRepos(apiRepos);
      // Keep selection if still present; otherwise reset.
      if (selectedRepo && !apiRepos.find((r) => r.repo_id === selectedRepo)) {
        setSelectedRepo("");
      }
      return apiRepos;
    } catch (error) {
      console.error(error);
    }
  }

  function extractRepoName(repo: RepoInfo): string {
    const full = repo.full_name || "";
    const name = full.includes("/") ? full.split("/").pop() || full : full;
    return (name || repo.repo_id).replace(/\.git$/i, "");
  }

  function handleSearch(term: string) {
    setSearchTerm(term);
  }

  async function handleDelete(repoId: string) {
    if (!session) return;
    setDeleteBusy(repoId);
    try {
      await deleteRepo(session.session_token, repoId);
      if (selectedRepo === repoId) setSelectedRepo("");
      await refreshRepos();
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setDeleteBusy(null);
    }
  }

  async function handleIndex(targetRepoId?: string, opts?: { force?: boolean }) {
    if (!session) {
      setStatus("Login first");
      return;
    }
    const repoId = targetRepoId ?? selectedRepo;
    if (!repoId) {
      setStatus("Choose a repo");
      return;
    }
    const meta = repos.find((r) => r.repo_id === repoId);
    if (!meta) {
      setStatus("Unknown repository");
      return;
    }
    if (!meta.repo_path) {
      setStatus("Repository is not available on disk yet.");
      return;
    }

    try {
      setIndexing(true);
      const forceReindex = opts?.force ?? true;
      setStatus(`${forceReindex ? "Re-indexing" : "Indexing"} ${meta.full_name || repoId}…`);
      const result = await indexRepo(session.session_token, {
        repo_id: repoId,
        repo_url: meta.repo_url || undefined,
        branch: meta.default_branch || undefined,
        force_reindex: forceReindex,
      });
      setStatus(result.status || "Indexing complete.");
      await refreshRepos();
    } catch (error) {
      setStatus((error as Error).message);
      await refreshRepos();
      if (!(opts?.force ?? true)) {
        setSelectedRepo("");
      }
    } finally {
      setIndexing(false);
    }
  }

  async function handleAsk(promptText?: string) {
    if (!session || !selectedRepo) return;
    const askText = (promptText ?? question).trim();
    if (!askText) return;
    const meta = repos.find((r) => r.repo_id === selectedRepo);
    if (!meta || !meta.indexed) {
      setStatus("Index this repository before asking questions.");
      return;
    }
    try {
      setLoading(true);
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: askText,
      };
      setMessages((prev) => [...prev, userMessage]);
      const answer: AskResponse = await askQuestion(session.session_token, {
        repo_id: selectedRepo,
        question: askText,
        top_k: 5,
        ensure_index: false,
        repo_url: meta?.repo_url || undefined,
        branch: meta?.default_branch || undefined,
      });
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: answer.answer,
        sources: answer.used_chunks,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setQuestion("");
    } catch (error) {
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: (error as Error).message,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setLoading(false);
    }
  }

  async function handleRemoteSelect(repo: RepoInfo) {
    if (!session) return;
    try {
      setBrowseOpen(false);
      setIndexing(true);
      setStatus(`Indexing ${repo.full_name || repo.repo_id}…`);
      const result = await indexRepo(session.session_token, {
        repo_id: repo.repo_id,
        repo_url: repo.repo_url || undefined,
        branch: repo.default_branch || undefined,
        force_reindex: true,
      });
      setSelectedRepo(repo.repo_id);
      setStatus(result.status || "Indexing complete.");
      await refreshRepos();
    } catch (error) {
      setStatus((error as Error).message);
    } finally {
      setIndexing(false);
    }
  }

  const selectedMeta = repos.find((repo) => repo.repo_id === selectedRepo);
  const chatDisabledReason = (() => {
    if (!selectedRepo) return "Select a repository to start chatting.";
    if (indexing) return "Indexing repository…";
    if (selectedMeta && !selectedMeta.indexed) return "Indexing repository…";
    return null;
  })();

  function handleRepoSelect(repoId: string) {
    setSelectedRepo(repoId);
    const meta = repos.find((r) => r.repo_id === repoId);
    if (!meta || indexing) return;
    if (!meta.indexed && meta.repo_path) {
      void handleIndex(repoId, { force: false });
    }
  }

  const showWorkspace = Boolean(session);
  const modalRepos =
    browseOpen && (filteredRepos.length || searchTerm)
      ? filteredRepos
      : browseOpen
        ? remoteRepos
        : [];

  return (
    <div className="app-shell">
      <main className="app-main">
        {!showWorkspace ? (
          <LoginPanel />
        ) : (
          <div className="workspace">
            <RepoToolbar
              repos={localRepos}
              selectedRepo={selectedRepo}
              busy={indexing}
              status={status}
              onSelect={handleRepoSelect}
              onRefresh={refreshRepos}
              onOpenBrowse={() => setBrowseOpen(true)}
              onDelete={handleDelete}
              deleteBusy={deleteBusy}
            />
            <div className="chat-shell">
              <ChatPanel
                repoId={selectedRepo}
                question={question}
                messages={messages}
                loading={loading}
                disabledReason={chatDisabledReason}
                onQuestionChange={setQuestion}
                onAsk={handleAsk}
              />
            </div>
          </div>
        )}
      </main>
      {browseOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-card stack-sm">
            <div className="modal-header">
              <h3>Browse remote repositories</h3>
              <button type="button" className="icon-btn" onClick={() => setBrowseOpen(false)}>
                ×
              </button>
            </div>
            <input
              value={searchTerm}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="Search by repository name"
            />
            <div className="browse-list">
              {modalRepos.length === 0 ? (
                <div className="combo-empty">
                  {remoteRepos.length === 0
                    ? "No remote repositories found. Trigger a GitHub sync first."
                    : `No matches for “${searchTerm}”.`}
                </div>
              ) : (
                modalRepos.map((repo) => (
                  <button
                    key={repo.repo_id}
                    type="button"
                    className="browse-item"
                    onClick={() => handleRemoteSelect(repo)}
                  >
                    <div className="combo-title">{extractRepoName(repo)}</div>
                    <div className="combo-sub">
                      <span className="mono small">{repo.repo_url}</span>
                      {repo.default_branch && <span className="pill subtle">Branch: {repo.default_branch}</span>}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
