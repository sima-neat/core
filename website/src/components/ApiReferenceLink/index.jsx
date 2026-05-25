import React, {useEffect, useMemo, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {trackDocsEvent} from '@site/src/lib/analytics';
import styles from './styles.module.css';

const LANG_PREF_KEY = 'neat-docs-language';
const PYTHON_SLUG_MAP = {
  'simaai-neat-tensor': 'Tensor',
  'simaai-neat-sample': 'Sample',
  'simaai-neat-session': 'Session',
  'simaai-neat-model': 'Model',
  'simaai-neat-pipelinerun': 'Run',
  'simaai-neat-node': 'Node',
  'simaai-neat-nodegroup': 'NodeGroup',
  'simaai-neat-runoptions': 'RunOptions',
  'simaai-neat-outputtensoroptions': 'OutputTensorOptions',
  'simaai-neat-inputoptions': 'InputOptions',
  'simaai-neat-outputoptions': 'OutputOptions',
};

function getPreferredLang() {
  if (typeof window === 'undefined') return 'cpp';
  return window.localStorage.getItem(LANG_PREF_KEY) || 'cpp';
}

function slugToPythonClass(slug) {
  if (PYTHON_SLUG_MAP[slug]) return PYTHON_SLUG_MAP[slug];
  let base = slug || '';
  if (base.startsWith('simaai-neat-')) base = base.replace('simaai-neat-', '');
  const parts = base.split(/[-_/]+/).filter(Boolean);
  if (parts.length === 0) return null;
  return parts.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join('');
}

function resolvePythonHref(href) {
  if (typeof href !== 'string') return href;
  if (href.includes('/reference/pythonapi/modules/')) return href;
  const match = href.match(/\/reference\/pythonapi\/(classes|structs|namespaces|files)\/([^#?]+)/);
  if (!match) return href;
  const slug = match[2];
  const klass = slugToPythonClass(slug);
  if (!klass) return '/reference/pythonapi/modules/pyneat';
  return `/reference/pythonapi/modules/pyneat/${klass}`;
}

function resolveApiHref(href, preferredLang = 'cpp') {
  if (typeof href !== 'string') return href;
  const segment = preferredLang === 'py' ? 'pythonapi' : 'cppapi';
  const resolved = href
    .replace('/reference/%7Blsa%7D/', '/reference/{lsa}/')
    .replace('/reference/{lsa}/', `/reference/${segment}/`);
  if (segment === 'pythonapi') {
    return resolvePythonHref(resolved);
  }
  return resolved;
}

function isApiReferenceHref(href) {
  if (typeof href !== 'string') return false;
  return (
    href.startsWith('/reference/cppapi/') ||
    href.startsWith('/reference/pythonapi/') ||
    href.includes('/reference/{lsa}/') ||
    href.includes('/reference/%7Blsa%7D/')
  );
}

function isInternalRootHref(href) {
  return typeof href === 'string' && href.startsWith('/') && !href.startsWith('//');
}

export default function ApiReferenceLink(props) {
  const {href, children, onClick, ...rest} = props;
  const [open, setOpen] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const [preferredLang, setPreferredLang] = useState('cpp');

  useEffect(() => {
    const handler = () => setPreferredLang(getPreferredLang());
    handler();
    window.addEventListener('neat:langchange', handler);
    window.addEventListener('storage', handler);
    return () => {
      window.removeEventListener('neat:langchange', handler);
      window.removeEventListener('storage', handler);
    };
  }, []);

  const resolvedHref = useMemo(() => resolveApiHref(href, preferredLang), [href, preferredLang]);
  const baseHref = useBaseUrl(isInternalRootHref(href) ? href : '');
  const resolvedBaseHref = useBaseUrl(isInternalRootHref(resolvedHref) ? resolvedHref : '');
  const linkHref = isInternalRootHref(href) ? baseHref : href;
  const resolvedLinkHref = isInternalRootHref(resolvedHref) ? resolvedBaseHref : resolvedHref;

  const title = useMemo(() => {
    if (typeof children === 'string') return children;
    return resolvedLinkHref;
  }, [children, resolvedLinkHref]);

  useEffect(() => {
    if (!open) return undefined;
    const handler = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    // Reset iframe when opening so onLoad runs consistently.
    setLoading(true);
    setIframeKey((v) => v + 1);
  }, [open, resolvedHref]);

  const handleIframeLoad = (event) => {
    const iframe = event.currentTarget;
    const doc = iframe?.contentDocument;
    if (!doc) return;

    const styleId = 'neat-api-popup-style';
    if (doc.getElementById(styleId)) return;

    const style = doc.createElement('style');
    style.id = styleId;
    style.textContent = `
      nav.navbar,
      header.navbar {
        display: none !important;
      }
      footer.footer,
      footer[class*="footer"] {
        display: none !important;
      }
      aside[class*="docSidebarContainer"],
      aside[class*="theme-doc-sidebar-container"],
      nav[class*="theme-doc-sidebar-menu"] {
        display: none !important;
      }
      .theme-doc-breadcrumbs,
      nav[aria-label="Breadcrumbs"],
      main h1,
      article h1 {
        display: none !important;
      }
      main[class*="docMainContainer"],
      main[class*="docMainContainer_"] {
        max-width: 100% !important;
        width: 100% !important;
      }
      div[class*="docPage"] {
        grid-template-columns: 1fr !important;
      }
    `;
    doc.head.appendChild(style);
    setLoading(false);
  };

  if (!isApiReferenceHref(href)) {
    return <a href={linkHref} onClick={onClick} {...rest}>{children}</a>;
  }

  const handleClick = (event) => {
    if (onClick) onClick(event);
    if (event.defaultPrevented) return;
    trackDocsEvent('api_reference_view', {
      language: resolvedHref.includes('/reference/pythonapi/') ? 'python' : 'cpp',
      link_url: resolvedLinkHref,
      source: 'inline_preview',
    });
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    setOpen(true);
  };

  return (
    <>
      <a href={resolvedLinkHref} onClick={handleClick} {...rest}>
        {children}
      </a>
      {open ? (
        <div className={styles.backdrop} onClick={() => setOpen(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.header}>
              <div className={styles.title}>{title}</div>
              <div className={styles.actions}>
                <a className={styles.openButton} href={resolvedLinkHref} target="_blank" rel="noreferrer">
                  Open in new tab
                </a>
                <button className={styles.close} type="button" onClick={() => setOpen(false)}>
                  Close
                </button>
              </div>
            </div>
            <div className={styles.body}>
              {loading ? (
                <div className={styles.loadingOverlay}>
                  <div className={styles.spinner} />
                  <div className={styles.loadingText}>Loading API reference…</div>
                </div>
              ) : null}
              <iframe
                key={iframeKey}
                title={`API preview ${title}`}
                src={resolvedLinkHref}
                className={`${styles.iframe} ${loading ? styles.iframeHidden : ''}`}
                onLoad={handleIframeLoad}
              />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
