import type { RepoInfo } from "@/lib/types";

interface Props {
  repos: RepoInfo[];
  selectedRepo: string;
  selectedRepoUrl?: string | null;
  selectedBranch?: string | null;
  indexing: boolean;
  status?: string | null;
  onSelectExisting: (repoId: string) => void;
  onIndex: () => void;
  onRefresh: () => void;
}

export function RepoPanel({
  repos,
  selectedRepo,
  selectedRepoUrl,
  selectedBranch,
  indexing,
  status,
  onSelectExisting,
  onIndex,
  onRefresh,
}: Props) {
  return (
    <section className="panel stack-sm">
      <div className="panel-header">
        <h3>Repository</h3>
        <button className="ghost" type="button" onClick={onRefresh} disabled={indexing}>
          Refresh
        </button>
      </div>

      <label className="select-label">
        Existing repositories
        <div className="select-control">
          <select value={selectedRepo} onChange={(event) => onSelectExisting(event.target.value)}>
            <option value="">— Select indexed repo —</option>
            {repos.map((repo) => (
              <option key={repo.repo_id} value={repo.repo_id}>
                {repo.repo_id} {repo.indexed ? "(indexed)" : ""}
              </option>
            ))}
          </select>
          <span aria-hidden="true" className="select-caret" />
        </div>
      </label>

      {selectedRepo && (
        <div className="stack-xs">
          <div className="muted">Clone URL</div>
          <div className="mono small">{selectedRepoUrl || "—"}</div>
          <div className="muted">Branch</div>
          <div className="mono small">{selectedBranch || "—"}</div>
        </div>
      )}

      <button className="primary" type="button" onClick={onIndex} disabled={indexing || !selectedRepo}>
        {indexing ? "Indexing…" : "Index repository"}
      </button>

      {status && <p className="status-text">{status}</p>}
    </section>
  );
}
