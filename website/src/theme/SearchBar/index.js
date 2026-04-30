import React, {useEffect, useMemo, useRef, useState} from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import clsx from "clsx";
import {liteClient as algoliasearch} from "algoliasearch/lite";
import {trackDocsEvent} from "@site/src/lib/analytics";
import styles from "./styles.module.css";

const MAX_RESULTS = 24;
const GROUP_ORDER = [
  "Getting Started",
  "How-To",
  "References",
  "Tutorials",
  "Contribute",
  "Other",
];

function sectionFromUrl(urlPath) {
  if (urlPath.includes("/getting-started/")) return "Getting Started";
  if (urlPath.includes("/how-to/")) return "How-To";
  if (urlPath.includes("/reference/")) return "References";
  if (urlPath.includes("/tutorials/")) return "Tutorials";
  if (urlPath.includes("/contribute/")) return "Contribute";
  return "Other";
}

function hitTitle(hit) {
  const h1 = hit.hierarchy?.lvl0;
  const h2 = hit.hierarchy?.lvl1;
  const h3 = hit.hierarchy?.lvl2;
  return h3 || h2 || h1 || hit.title || hit.url || "Untitled";
}

function hitSnippet(hit) {
  const v = hit._snippetResult?.content?.value;
  if (v) return v;
  return hit.content || "";
}

function hitRoute(hitUrl) {
  if (!hitUrl || typeof hitUrl !== "string") return "/";
  if (hitUrl.startsWith("/")) return hitUrl;
  if (/^https?:\/\//i.test(hitUrl)) {
    try {
      const u = new URL(hitUrl);
      return `${u.pathname}${u.search}${u.hash}`;
    } catch {
      return hitUrl;
    }
  }
  return hitUrl;
}

function hitRouteWithQuery(hitUrl, q) {
  const route = hitRoute(hitUrl);
  const term = (q || "").trim();
  if (!term) return route;
  try {
    const u = new URL(route, "https://local.search");
    u.searchParams.set("q", term);
    return `${u.pathname}${u.search}${u.hash}`;
  } catch {
    return route;
  }
}

export default function SearchBar() {
  const {siteConfig} = useDocusaurusContext();
  const appId = siteConfig.customFields?.algoliaSearch?.appId;
  const apiKey = siteConfig.customFields?.algoliaSearch?.apiKey;
  const indexName = siteConfig.customFields?.algoliaSearch?.indexName;
  const enabled = Boolean(
    appId &&
      apiKey &&
      indexName &&
      appId !== "REPLACE_ME" &&
      apiKey !== "REPLACE_ME" &&
      indexName !== "REPLACE_ME",
  );

  const searchClient = useMemo(() => {
    if (!enabled) return null;
    return algoliasearch(appId, apiKey);
  }, [enabled, appId, apiKey]);

  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hits, setHits] = useState([]);
  const [activeGroup, setActiveGroup] = useState("");
  const wrapperRef = useRef(null);
  const inputRef = useRef(null);
  const lastTrackedSearchRef = useRef("");

  useEffect(() => {
    if (!open) return undefined;
    function onClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  useEffect(() => {
    function onKeyDown(event) {
      const isFindShortcut =
        event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey) && !event.altKey;
      if (!isFindShortcut) return;

      const target = event.target;
      const tag = target?.tagName?.toLowerCase?.() || "";
      const inEditable =
        tag === "input" || tag === "textarea" || target?.isContentEditable === true;
      if (inEditable) return;

      event.preventDefault();
      const el = inputRef.current;
      if (!el) return;
      el.focus();
      el.select();
      setOpen(query.trim().length > 0);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [query]);

  useEffect(() => {
    if (!enabled || !searchClient) return undefined;
    const trimmed = query.trim();
    if (!trimmed) {
      setHits([]);
      setLoading(false);
      setError("");
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    const t = setTimeout(async () => {
      try {
        const response = await searchClient.search([
          {
            indexName,
            query: trimmed,
            params: {
              hitsPerPage: MAX_RESULTS,
              attributesToSnippet: ["content:24"],
            },
          },
        ]);
        if (cancelled) return;
        const found = response.results?.[0]?.hits || [];
        setHits(found);
        if (trimmed.length >= 2 && lastTrackedSearchRef.current !== trimmed) {
          lastTrackedSearchRef.current = trimmed;
          trackDocsEvent("search", {
            search_term: trimmed,
            search_location: "navbar",
            result_count: found.length,
          });
        }
        setActiveGroup((prev) => {
          if (!found.length) return "";
          const available = new Set(found.map((h) => sectionFromUrl(h.url || "")));
          if (prev && available.has(prev)) return prev;
          for (const key of GROUP_ORDER) {
            if (available.has(key)) return key;
          }
          return "";
        });
      } catch (err) {
        if (!cancelled) {
          setError(err?.message || "Search request failed.");
          setHits([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 140);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [query, enabled, indexName, searchClient]);

  const groupedHits = useMemo(() => {
    const groups = new Map();
    for (const hit of hits) {
      const section = sectionFromUrl(hit.url || "");
      if (!groups.has(section)) groups.set(section, []);
      groups.get(section).push(hit);
    }
    return GROUP_ORDER.filter((k) => groups.has(k)).map((k) => [k, groups.get(k)]);
  }, [hits]);
  const activeHits = useMemo(() => {
    if (!groupedHits.length) return [];
    if (!activeGroup) return groupedHits[0][1];
    const found = groupedHits.find(([group]) => group === activeGroup);
    return found ? found[1] : groupedHits[0][1];
  }, [groupedHits, activeGroup]);

  if (!enabled) {
    return (
      <div className={styles.searchDisabled} title="Set DOCS_ALGOLIA_* env vars to enable search.">
        Search
      </div>
    );
  }

  return (
    <div ref={wrapperRef} className={styles.searchRoot}>
      <input
        ref={inputRef}
        type="search"
        className={styles.searchInput}
        placeholder="Search docs…"
        value={query}
        onFocus={() => {
          if (query.trim().length > 0) {
            setOpen(true);
          }
        }}
        onChange={(e) => {
          const next = e.target.value;
          setQuery(next);
          setOpen(next.trim().length > 0);
        }}
        aria-label="Search documentation"
      />

      {open && (
        <div className={styles.searchPanel}>
          {loading && <div className={styles.status}>Searching…</div>}
          {!loading && error && <div className={styles.error}>{error}</div>}
          {!loading && !error && query.trim() && hits.length === 0 && (
            <div className={styles.status}>No matches found.</div>
          )}
          {!loading && !error && !query.trim() && (
            <div className={styles.status}>Try terms like "session", "build", or "RunStats".</div>
          )}

          {!loading && !error && groupedHits.length > 0 && (
            <div className={styles.layout}>
              <aside className={styles.categoryPane}>
                <ul className={styles.categoryList}>
                  {groupedHits.map(([group, groupHits]) => {
                    const selected = (activeGroup || groupedHits[0][0]) === group;
                    return (
                      <li key={group}>
                        <button
                          type="button"
                          className={clsx(styles.categoryButton, selected && styles.categoryButtonActive)}
                          onClick={() => setActiveGroup(group)}
                        >
                          <span>{group}</span>
                          <span className={styles.categoryCount}>({groupHits.length})</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </aside>

              <section className={styles.resultsPane}>
                <ul className={styles.resultList}>
                  {activeHits.map((hit) => (
                    <li key={hit.objectID} className={styles.resultItem}>
                      <Link
                        to={hitRouteWithQuery(hit.url, query)}
                        className={styles.resultLink}
                        onClick={() => {
                          const rank = activeHits.findIndex((item) => item.objectID === hit.objectID) + 1;
                          trackDocsEvent("docs_search_result_click", {
                            search_term: query,
                            search_location: "navbar",
                            result_rank: rank,
                            result_section: sectionFromUrl(hit.url || ""),
                            result_url: hitRoute(hit.url),
                          });
                          setOpen(false);
                        }}
                      >
                        <div
                          className={styles.resultTitle}
                          dangerouslySetInnerHTML={{__html: hitTitle(hit)}}
                        />
                        <div
                          className={clsx(styles.resultSnippet, "algolia-search-snippet")}
                          dangerouslySetInnerHTML={{__html: hitSnippet(hit)}}
                        />
                      </Link>
                    </li>
                  ))}
                </ul>
              </section>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
