const ROUTE_EVENT = "sima-search-highlight-route";
let historyPatched = false;

function getQueryTerm() {
  const params = new URLSearchParams(window.location.search || "");
  const q = (params.get("q") || "").trim();
  return q;
}

function highlightInDocs() {
  const term = getQueryTerm();
  const containers = Array.from(
    document.querySelectorAll(".theme-doc-markdown, article, main"),
  ).filter((el) => el && el.textContent && el.textContent.trim().length > 0);

  if (!containers.length) return;

  const clearHighlights = (container) => {
    const marked = container.querySelectorAll("mark.docSearchHighlight");
    marked.forEach((m) => {
      const t = document.createTextNode(m.textContent || "");
      m.replaceWith(t);
    });
    container.normalize();
  };

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const markTerm = (container, query) => {
    if (!query) return;
    const rx = new RegExp(escapeRegExp(query), "gi");
    const walker = document.createTreeWalker(
      container,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          const text = node.nodeValue || "";
          if (!text.trim()) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (
            parent.closest(
              "pre, code, kbd, script, style, .navbar, .menu, button, a[role='button'], mark.docSearchHighlight",
            )
          ) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      },
    );

    const nodes = [];
    while (walker.nextNode()) {
      nodes.push(walker.currentNode);
    }

    nodes.forEach((node) => {
      const text = node.nodeValue || "";
      rx.lastIndex = 0;
      if (!rx.test(text)) return;

      const frag = document.createDocumentFragment();
      let cursor = 0;
      rx.lastIndex = 0;
      let m;
      while ((m = rx.exec(text)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        if (start > cursor) {
          frag.appendChild(document.createTextNode(text.slice(cursor, start)));
        }
        const mark = document.createElement("mark");
        mark.className = "docSearchHighlight";
        mark.textContent = text.slice(start, end);
        frag.appendChild(mark);
        cursor = end;
      }
      if (cursor < text.length) {
        frag.appendChild(document.createTextNode(text.slice(cursor)));
      }
      node.replaceWith(frag);
    });
  };

  containers.forEach((container) => {
    clearHighlights(container);
    markTerm(container, term);
  });
}

function scheduleHighlight() {
  // Wait for Docusaurus route content render.
  window.setTimeout(highlightInDocs, 120);
}

function patchHistory() {
  if (historyPatched) return;
  historyPatched = true;

  const origPushState = window.history.pushState;
  const origReplaceState = window.history.replaceState;

  window.history.pushState = function patchedPushState(...args) {
    const ret = origPushState.apply(this, args);
    window.dispatchEvent(new Event(ROUTE_EVENT));
    return ret;
  };

  window.history.replaceState = function patchedReplaceState(...args) {
    const ret = origReplaceState.apply(this, args);
    window.dispatchEvent(new Event(ROUTE_EVENT));
    return ret;
  };
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  patchHistory();
  window.addEventListener("popstate", scheduleHighlight);
  window.addEventListener(ROUTE_EVENT, scheduleHighlight);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleHighlight, {once: true});
  } else {
    scheduleHighlight();
  }
}
