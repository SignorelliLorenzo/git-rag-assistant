import { useEffect, useRef, useState } from "react";
import type { RepoInfo } from "@/lib/types";

interface Props {
  repos: RepoInfo[];
  selectedRepo: string;
  busy: boolean;
  status?: string | null;
  onSelect: (repoId: string) => void;
  onRefresh: () => void;
  onOpenBrowse: () => void;
  onDelete: (repoId: string) => void;
  deleteBusy?: string | null;
}

export function RepoToolbar({
  repos,
  selectedRepo,
  busy,
  status,
  onSelect,
  onRefresh,
  onOpenBrowse,
  onDelete,
  deleteBusy,
}: Props) {
  const [open, setOpen] = useState(false);
  const comboRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handleClick(event: MouseEvent) {
      if (comboRef.current && !comboRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  useEffect(() => {
    if (busy) {
      setOpen(false);
    }
  }, [busy]);

  function handleSelect(repoId: string) {
    onSelect(repoId);
    setOpen(false);
  }

  function renderLabel() {
    if (!selectedRepo) return "Select a repository";
    const meta = repos.find((r) => r.repo_id === selectedRepo);
    return meta?.full_name || selectedRepo;
  }

  return (
    <div className="repo-toolbar">
      <div className="repo-toolbar__group">
        <button
          type="button"
          className="icon-btn"
          aria-label="Refresh repositories"
          onClick={onRefresh}
          disabled={busy}
        >
          ↻
        </button>
        <div className="repo-combo">
          <div className="combo-caption">Repository</div>
          <div className={`combo-inline ${busy ? "is-disabled" : ""}`} ref={comboRef}>
            <button
              type="button"
              className="combo-trigger"
              onClick={() => setOpen((v: boolean) => !v)}
              disabled={busy}
            >
              <span className="combo-label">{renderLabel()}</span>
              <span aria-hidden className="select-caret" />
            </button>
            {open && (
              <div className="combo-menu">
                {repos.length === 0 && <div className="combo-empty">No repositories indexed yet.</div>}
                {repos.map((repo) => (
                  <div key={repo.repo_id} className="combo-item">
                    <button
                      type="button"
                      className="combo-select"
                      onClick={() => handleSelect(repo.repo_id)}
                    >
                      <div className="combo-title">{repo.full_name || repo.repo_id}</div>
                    </button>
                    <button
                      type="button"
                      className="icon-btn danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(repo.repo_id);
                        setOpen(false);
                      }}
                      aria-label={`Delete ${repo.full_name || repo.repo_id}`}
                      disabled={deleteBusy === repo.repo_id || busy}
                    >
                      {deleteBusy === repo.repo_id ? "…" : "×"}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="repo-toolbar__actions">
        <button type="button" className="icon-btn" onClick={onOpenBrowse} disabled={busy}>
          +
        </button>
        {status && <span className="repo-toolbar__status">{status}</span>}
      </div>
    </div>
  );
}
