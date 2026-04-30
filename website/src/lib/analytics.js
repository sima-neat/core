export function trackDocsEvent(name, params = {}) {
  if (typeof window === "undefined" || !name) return;

  const detail = {name, params};
  if (typeof window.neatDocsTrack === "function") {
    window.neatDocsTrack(name, params);
    return;
  }

  window.dispatchEvent(new CustomEvent("neat:analytics-track", {detail}));
}
