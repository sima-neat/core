import React, {useEffect, useMemo, useState} from 'react';
import styles from './styles.module.css';

const DEFAULT_ORG = 'manuel-roldan';
const DEFAULT_REPO = 'PipelineSession';
const DEFAULT_BRANCH = 'main';

function buildGitHubUrl({org, repo, branch, path}) {
  return `https://github.com/${org}/${repo}/blob/${branch}/${path}`;
}

function buildRawUrl({org, repo, branch, path}) {
  return `https://raw.githubusercontent.com/${org}/${repo}/${branch}/${path}`;
}

export default function CodePopup({
  path,
  org = DEFAULT_ORG,
  repo = DEFAULT_REPO,
  branch = DEFAULT_BRANCH,
  label = 'View source code',
}) {
  const [open, setOpen] = useState(false);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const ghUrl = useMemo(() => buildGitHubUrl({org, repo, branch, path}), [org, repo, branch, path]);
  const rawUrl = useMemo(() => buildRawUrl({org, repo, branch, path}), [org, repo, branch, path]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError('');
    setContent('');
    fetch(rawUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((text) => {
        if (cancelled) return;
        setContent(text);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(`Failed to load source (${err.message}).`);
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, rawUrl]);

  return (
    <>
      <button className={styles.trigger} type="button" onClick={() => setOpen(true)}>
        {label}
      </button>
      <a className={styles.link} href={ghUrl} target="_blank" rel="noreferrer">
        Open on GitHub
      </a>
      {open ? (
        <div className={styles.backdrop} onClick={() => setOpen(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.header}>
              <div className={styles.title}>{path}</div>
              <button className={styles.close} type="button" onClick={() => setOpen(false)}>
                Close
              </button>
            </div>
            <div className={styles.body}>
              {loading ? (
                <div className={styles.loading}>Loading…</div>
              ) : error ? (
                <div className={styles.error}>{error}</div>
              ) : (
                <pre className={styles.code}>
                  <code>{content}</code>
                </pre>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
