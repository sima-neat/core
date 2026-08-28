const PREVIEW_PARAM = "i18n";
const PREVIEW_SESSION_KEY = "neat-docs-i18n-preview";
const LOCALIZED_LOCALES = new Set(["ko", "ja", "zh-Hant", "uk"]);

function previewEnabled() {
  const enabledByUrl = new URLSearchParams(window.location.search).get(PREVIEW_PARAM) === "1";
  if (enabledByUrl) {
    try {
      window.sessionStorage.setItem(PREVIEW_SESSION_KEY, "1");
    } catch {
      // Session storage may be unavailable in privacy-restricted contexts.
    }
    return true;
  }

  try {
    return window.sessionStorage.getItem(PREVIEW_SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

function currentLocale() {
  return window.location.pathname
    .split("/")
    .filter(Boolean)
    .find((segment) => LOCALIZED_LOCALES.has(segment));
}

function preferredLocale() {
  if (LOCALIZED_LOCALES.has(window.__NEAT_DOCS_PREFERRED_LOCALE__)) {
    return window.__NEAT_DOCS_PREFERRED_LOCALE__;
  }
  const entry = String(document.cookie || "")
    .split("; ")
    .find((cookie) => cookie.startsWith("sima-neat-locale="));
  let locale = "";
  try {
    locale = entry ? decodeURIComponent(entry.split("=").slice(1).join("=")) : "";
  } catch {
    // Treat malformed externally written cookie values as no preference.
  }
  return LOCALIZED_LOCALES.has(locale) ? locale : "";
}

function localizationEnabled() {
  return previewEnabled() || preferredLocale() === currentLocale();
}

function isStaticFileUrl(url) {
  const segments = url.pathname.split("/").filter(Boolean);
  const lastSegment = segments[segments.length - 1] || "";
  return lastSegment.includes(".");
}

function retainCurrentLocale(url) {
  const currentSegments = window.location.pathname.split("/").filter(Boolean);
  const localeIndex = currentSegments.findIndex((segment) => LOCALIZED_LOCALES.has(segment));
  if (localeIndex < 0) return url;

  const targetSegments = url.pathname.split("/").filter(Boolean);
  if (targetSegments.some((segment) => LOCALIZED_LOCALES.has(segment))) return url;

  const baseSegments = currentSegments.slice(0, localeIndex);
  const targetHasBase = baseSegments.every((segment, index) => targetSegments[index] === segment);
  if (!targetHasBase) return url;

  targetSegments.splice(baseSegments.length, 0, currentSegments[localeIndex]);
  url.pathname = `/${targetSegments.join("/")}${url.pathname.endsWith("/") ? "/" : ""}`;
  return url;
}

function previewUrl(rawHref) {
  const url = new URL(rawHref, window.location.href);
  if (url.origin !== window.location.origin) return null;
  // Static directories are shared by every locale and are not emitted below
  // locale-prefixed routes. Leave file/download URLs on that shared root.
  if (isStaticFileUrl(url)) return null;
  retainCurrentLocale(url);
  if (previewEnabled()) url.searchParams.set(PREVIEW_PARAM, "1");
  return url;
}

function retainPreviewFlag() {
  if (!localizationEnabled()) return;

  document.querySelectorAll("a[href]").forEach((anchor) => {
    const rawHref = anchor.getAttribute("href");
    if (!rawHref || rawHref.startsWith("#")) return;
    if (anchor.hasAttribute?.("download")) return;

    try {
      const url = previewUrl(rawHref);
      if (!url) return;
      anchor.setAttribute("href", `${url.pathname}${url.search}${url.hash}`);
    } catch {
      // Ignore non-URL href values such as mailto: and custom protocols.
    }
  });
}

function retainPreviewFlagForClick(event) {
  if (!localizationEnabled()) return;
  if (
    (event.button != null && event.button !== 0) ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) return;

  const anchor = event.target.closest?.("a[href]");
  if (!anchor) return;

  const rawHref = anchor.getAttribute("href");
  if (!rawHref || rawHref.startsWith("#")) return;
  const target = anchor.getAttribute("target");
  if ((target && target !== "_self") || anchor.hasAttribute?.("download")) return;

  try {
    const url = previewUrl(rawHref);
    if (!url) return;
    const destination = `${url.pathname}${url.search}${url.hash}`;
    anchor.setAttribute("href", destination);

    // Docusaurus' router handles Link clicks from the original React `to`
    // prop, so changing the DOM href alone does not preserve locale or preview
    // state. Own unmodified internal clicks while localization is active.
    event.preventDefault();
    event.stopImmediatePropagation?.();
    event.stopPropagation();
    window.location.assign(destination);
  } catch {
    // Ignore non-URL href values such as mailto: and custom protocols.
  }
}

if (typeof document !== "undefined") {
  retainPreviewFlag();
  const observer = new MutationObserver(retainPreviewFlag);
  observer.observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener("click", retainPreviewFlagForClick, true);
}
