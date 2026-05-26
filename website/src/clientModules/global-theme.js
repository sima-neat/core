const THEME_COOKIE = "sima-neat-theme";
const THEME_KEYS = ["theme", "portal-theme"];
const VALID_THEMES = new Set(["light", "dark"]);

function normalizeTheme(value) {
  return VALID_THEMES.has(value) ? value : null;
}

function readCookieTheme() {
  const entry = document.cookie
    .split("; ")
    .find((cookie) => cookie.startsWith(`${THEME_COOKIE}=`));
  return normalizeTheme(entry ? decodeURIComponent(entry.split("=").slice(1).join("=")) : null);
}

function cookieDomain() {
  const { hostname } = window.location;
  if (hostname === "localhost" || /^[\d.]+$/.test(hostname)) {
    return "";
  }

  if (hostname.endsWith(".neat.sima.ai") || hostname.endsWith(".neat.paconsultings.com")) {
    const parts = hostname.split(".");
    return parts.length > 3 ? parts.slice(1).join(".") : hostname;
  }

  return "";
}

function writeCookieTheme(theme) {
  const domain = cookieDomain();
  const domainPart = domain ? `; Domain=.${domain}` : "";
  const securePart = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `${THEME_COOKIE}=${encodeURIComponent(theme)}; Path=/; Max-Age=31536000; SameSite=Lax${securePart}${domainPart}`;
}

function writeLocalTheme(theme) {
  THEME_KEYS.forEach((key) => {
    try {
      window.localStorage.setItem(key, theme);
    } catch (_) {
      // Ignore private browsing and restricted storage modes.
    }
  });
}

function applyTheme(theme) {
  if (!theme) return;
  document.documentElement.setAttribute("data-theme", theme);
  document.documentElement.setAttribute("data-theme-choice", theme);
  writeLocalTheme(theme);
}

function persistTheme(theme) {
  if (!theme) return;
  writeLocalTheme(theme);
  writeCookieTheme(theme);
}

function syncFromCookie() {
  const cookieTheme = readCookieTheme();
  if (cookieTheme && cookieTheme !== document.documentElement.getAttribute("data-theme")) {
    applyTheme(cookieTheme);
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  applyTheme(readCookieTheme());

  let lastTheme = normalizeTheme(document.documentElement.getAttribute("data-theme"));
  const observer = new MutationObserver(() => {
    const nextTheme = normalizeTheme(document.documentElement.getAttribute("data-theme"));
    if (nextTheme && nextTheme !== lastTheme) {
      lastTheme = nextTheme;
      persistTheme(nextTheme);
    }
  });

  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  window.addEventListener("focus", syncFromCookie);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      syncFromCookie();
    }
  });
}
