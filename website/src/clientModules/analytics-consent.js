import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";

const CONSENT_KEY = "sima-developer-center-cookie-consent";
const CONSENT_VERSION = 1;
const ANALYTICS_EVENT = "neat:analytics-consent";
const TRACK_EVENT = "neat:analytics-track";
const INTERNAL_HOSTS = new Set([
  "developer.sima.ai",
  "build.stg.neat.sima.ai",
  "localhost",
  "127.0.0.1",
]);
const DOWNLOAD_EXTENSIONS = new Set([
  "deb",
  "gz",
  "tgz",
  "whl",
  "zip",
  "pdf",
]);

let gtagLoaded = false;
let initialPageViewSent = false;
let lastTrackedLocation = "";
let lastDerivedLocation = "";
let initialized = false;

const deniedConsent = {
  analytics_storage: "denied",
  ad_storage: "denied",
  ad_user_data: "denied",
  ad_personalization: "denied",
};

const analyticsGrantedConsent = {
  ...deniedConsent,
  analytics_storage: "granted",
};

const getAnalyticsConfig = () => window.__NEAT_DOCS_ANALYTICS__ || {};

const getCurrentPath = () => `${window.location.pathname}${window.location.search}`;

const cleanString = (value, maxLength = 160) => {
  if (value === undefined || value === null) return undefined;
  const normalized = String(value).replace(/\s+/g, " ").trim();
  if (!normalized) return undefined;
  return normalized.slice(0, maxLength);
};

const sanitizeSearchTerm = (value) => {
  const term = cleanString(value, 80);
  if (!term) return undefined;
  if (/@/.test(term)) return "[redacted]";
  return term;
};

const sanitizeUrl = (value) => {
  if (!value) return undefined;
  try {
    const url = new URL(value, window.location.href);
    url.username = "";
    url.password = "";
    url.search = "";
    return url.toString().slice(0, 240);
  } catch {
    return cleanString(value, 240);
  }
};

const docSectionFromPath = (pathname) => {
  const parts = pathname.split("/").filter(Boolean);
  if (pathname.startsWith("/reference/cppapi")) return "api-reference-cpp";
  if (pathname.startsWith("/reference/pythonapi")) return "api-reference-python";
  if (parts[0] === "getting-started") return "getting-started";
  if (parts[0] === "tutorials") return "tutorials";
  if (parts[0] === "reference") return "reference";
  if (parts[0] === "contribute") return "contribute";
  return parts[0] || "home";
};

const getDocContext = () => {
  const pathname = window.location.pathname;
  const docId = pathname.replace(/^\/+|\/+$/g, "") || "index";
  const title = document.title.replace(/\s*\|\s*SiMa\.ai Neat\s*$/i, "");
  return {
    doc_id: docId,
    doc_title: cleanString(title, 120),
    doc_section: docSectionFromPath(pathname),
    page_path: getCurrentPath(),
  };
};

const normalizeParams = (params = {}) => {
  const normalized = {};
  for (const [key, value] of Object.entries(params || {})) {
    if (value === undefined || value === null || value === "") continue;
    if (key === "link_url") {
      normalized[key] = sanitizeUrl(value);
    } else if (key === "search_term") {
      normalized[key] = sanitizeSearchTerm(value);
    } else if (typeof value === "number" || typeof value === "boolean") {
      normalized[key] = value;
    } else {
      normalized[key] = cleanString(value);
    }
  }
  return normalized;
};

const trackEvent = (name, params = {}) => {
  if (!name || !window.gtag || !getStoredConsent()?.analytics) return;
  window.gtag("event", name, {
    ...getDocContext(),
    ...normalizeParams(params),
  });
};

const tutorialIdFromPath = (pathname) => {
  const match = pathname.match(/^\/tutorials\/([^/?#]+)/);
  return match ? match[1] : "";
};

const getSessionFlag = (key) => {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
};

const setSessionFlag = (key, value) => {
  try {
    window.sessionStorage.setItem(key, value);
  } catch {
    // Session storage may be unavailable in strict privacy modes.
  }
};

const trackRouteDerivedEvents = () => {
  const pagePath = getCurrentPath();
  if (pagePath === lastDerivedLocation) return;
  lastDerivedLocation = pagePath;

  const pathname = window.location.pathname.replace(/\/$/, "") || "/";
  if (/^\/reference\/(cppapi|pythonapi)(\/|$)/.test(pathname)) {
    trackEvent("api_reference_view", {
      language: pathname.includes("/pythonapi/") ? "python" : "cpp",
      source: "page_view",
    });
  }

  const tutorialId = tutorialIdFromPath(pathname);
  if (tutorialId) {
    const key = `neat-docs-tutorial-begin:${tutorialId}`;
    if (getSessionFlag(key) !== "1") {
      setSessionFlag(key, "1");
      trackEvent("tutorial_begin", {tutorial_id: tutorialId});
    }
  }

  if (pathname === "/getting-started/neat-library") {
    trackEvent("install_page_view", {
      platform: "neat-library",
    });
  }
};

const getStoredConsent = () => {
  try {
    const value = window.localStorage.getItem(CONSENT_KEY);
    const parsed = value ? JSON.parse(value) : null;
    if (!parsed || parsed.version !== CONSENT_VERSION) return null;
    return parsed;
  } catch {
    return null;
  }
};

const setStoredConsent = (consent) => {
  const next = {
    version: CONSENT_VERSION,
    analytics: Boolean(consent.analytics),
    marketing: false,
    updatedAt: new Date().toISOString(),
  };
  try {
    window.localStorage.setItem(CONSENT_KEY, JSON.stringify(next));
  } catch {
    // Storage can be unavailable in strict privacy modes.
  }
  window.dispatchEvent(new CustomEvent(ANALYTICS_EVENT, {detail: next}));
  return next;
};

const ensureGtag = () => {
  window.dataLayer = window.dataLayer || [];
  if (!window.gtag) {
    window.gtag = function gtag() {
      window.dataLayer.push(arguments);
    };
  }
  window.gtag("consent", "default", deniedConsent);
};

const updateGtagConsent = (consent) => {
  ensureGtag();
  window.gtag(
    "consent",
    "update",
    consent.analytics ? analyticsGrantedConsent : deniedConsent,
  );
};

const trackPageView = () => {
  if (!window.gtag || !getStoredConsent()?.analytics) return;
  const pagePath = getCurrentPath();
  if (pagePath === lastTrackedLocation) return;
  lastTrackedLocation = pagePath;
  window.gtag("event", "page_view", {
    page_title: document.title,
    page_location: window.location.href,
    page_path: pagePath,
  });
  trackRouteDerivedEvents();
};

const loadGtag = () => {
  const {measurementId} = getAnalyticsConfig();
  if (!measurementId || gtagLoaded) return;

  ensureGtag();
  gtagLoaded = true;

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(
    measurementId,
  )}`;
  document.head.appendChild(script);

  window.gtag("js", new Date());
  window.gtag("config", measurementId, {
    anonymize_ip: true,
    send_page_view: false,
  });

  if (!initialPageViewSent) {
    initialPageViewSent = true;
    trackPageView();
  }
};

const applyConsent = (consent) => {
  updateGtagConsent(consent);
  if (consent.analytics) {
    loadGtag();
  }
};

const closestLink = (target) => {
  if (!target?.closest) return null;
  return target.closest("a[href]");
};

const linkText = (link) => cleanString(link?.textContent || link?.getAttribute("aria-label"), 120);

const linkUrl = (link) => {
  if (!link) return "";
  try {
    return new URL(link.getAttribute("href"), window.location.href).toString();
  } catch {
    return link.getAttribute("href") || "";
  }
};

const isInternalUrl = (urlValue) => {
  try {
    const url = new URL(urlValue, window.location.href);
    return url.origin === window.location.origin || INTERNAL_HOSTS.has(url.hostname);
  } catch {
    return true;
  }
};

const isDownloadUrl = (urlValue) => {
  try {
    const url = new URL(urlValue, window.location.href);
    const ext = url.pathname.split(".").pop()?.toLowerCase();
    return DOWNLOAD_EXTENSIONS.has(ext || "");
  } catch {
    return false;
  }
};

const platformFromText = (value) => {
  const text = String(value || "").toLowerCase();
  if (text.includes("devkit")) return "devkit";
  if (text.includes("elxr") || text.includes("sdk")) return "elxr_sdk";
  return undefined;
};

const commandIdFromCode = (value) => {
  const code = String(value || "").toLowerCase();
  if (code.includes("sima-cli install ghcr:sima-neat/sdk")) return "install_elxr_sdk";
  if (code.includes("sima-cli install sdk")) return "install_sdk";
  if (code.includes("sima-cli install")) return "sima_cli_install";
  return undefined;
};

const languageFromCodeElement = (element) => {
  const className = element?.className || "";
  const match = String(className).match(/language-([a-z0-9_+-]+)/i);
  return match ? match[1] : undefined;
};

const codeTextNearButton = (button) => {
  const container = button.closest("[class*='codeBlockContainer'], .theme-code-block, div");
  const code = container?.querySelector?.("pre code") || container?.querySelector?.("code");
  return {
    codeText: code?.textContent || "",
    language: languageFromCodeElement(code || container),
  };
};

const isCopyButton = (button) => {
  const label = `${button.getAttribute("aria-label") || ""} ${button.title || ""} ${
    button.textContent || ""
  }`.toLowerCase();
  return label.includes("copy");
};

const bindInteractionTracking = () => {
  if (document.body.dataset.analyticsInteractionsBound === "1") return;
  document.body.dataset.analyticsInteractionsBound = "1";

  document.addEventListener(
    "click",
    (event) => {
      const button = event.target?.closest?.("button");
      if (button && isCopyButton(button)) {
        const {codeText, language} = codeTextNearButton(button);
        if (!codeText.trim()) return;
        const commandId = commandIdFromCode(codeText);
        window.setTimeout(() => {
          const params = {
            language,
            command_id: commandId,
            platform: platformFromText(codeText),
          };
          trackEvent("copy_code", params);
          if (commandId) {
            trackEvent("install_command_copy", params);
          }
        }, 0);
      }

      const link = closestLink(event.target);
      if (!link) return;
      const href = linkUrl(link);
      const text = linkText(link);
      const params = {
        link_url: href,
        link_text: text,
        platform: platformFromText(`${text} ${href}`),
      };

      if (/^https?:\/\/github\.com\//i.test(href)) {
        trackEvent("github_link_click", params);
      }

      if (/install/i.test(`${text} ${href}`)) {
        trackEvent("install_cta_click", params);
      }

      if (!isInternalUrl(href) && isDownloadUrl(href)) {
        trackEvent("external_download_click", params);
      }
    },
    true,
  );
};

const closeConsentUi = () => {
  document.querySelectorAll(".cookie-consent").forEach((el) => el.remove());
};

const renderPreferences = () => {
  closeConsentUi();
  const stored = getStoredConsent();
  const analyticsEnabled = stored?.analytics ?? false;

  const panel = document.createElement("section");
  panel.className = "cookie-consent cookie-consent--panel";
  panel.setAttribute("aria-label", "Cookie preferences");
  panel.innerHTML = `
    <div class="cookie-consent__content">
      <p class="cookie-consent__eyebrow">Privacy preferences</p>
      <h2 class="cookie-consent__title">Cookie settings</h2>
      <p class="cookie-consent__text">Necessary cookies keep the docs site working. Analytics cookies help SiMa.ai understand aggregate docs usage. Marketing cookies are not used on this site today.</p>
      <label class="cookie-consent__toggle">
        <span>
          <strong>Analytics</strong>
          <small>Google Analytics 4, loaded only after consent.</small>
        </span>
        <input type="checkbox" ${analyticsEnabled ? "checked" : ""} />
      </label>
      <div class="cookie-consent__details">
        <h3>What is collected</h3>
        <p>When analytics is accepted, the site may collect aggregate documentation usage such as page views, navigation paths, and engagement events. The site should not send names, email addresses, account identifiers, or other personal content to analytics.</p>
        <h3>Your choices</h3>
        <p>You can accept or reject optional analytics cookies. The docs site continues to work either way, and you can reopen these settings from the footer.</p>
      </div>
      <p class="cookie-consent__note">Global Privacy Control is respected by keeping marketing disabled.</p>
    </div>
    <div class="cookie-consent__actions">
      <button type="button" class="button button--primary" data-cookie-save>Save settings</button>
      <button type="button" class="button button--secondary" data-cookie-reject>Reject optional cookies</button>
    </div>
  `;

  panel.querySelector("[data-cookie-save]").addEventListener("click", () => {
    const input = panel.querySelector("input[type='checkbox']");
    applyConsent(setStoredConsent({analytics: input.checked}));
    closeConsentUi();
  });
  panel.querySelector("[data-cookie-reject]").addEventListener("click", () => {
    applyConsent(setStoredConsent({analytics: false}));
    closeConsentUi();
  });

  document.body.appendChild(panel);
};

const renderBanner = () => {
  if (document.querySelector(".cookie-consent")) return;

  const banner = document.createElement("section");
  banner.className = "cookie-consent cookie-consent--banner";
  banner.setAttribute("aria-label", "Cookie notice");
  banner.innerHTML = `
    <div class="cookie-consent__content">
      <p class="cookie-consent__eyebrow">Privacy</p>
      <h2 class="cookie-consent__title">Help improve these docs</h2>
      <p class="cookie-consent__text">We use optional Google Analytics cookies to understand aggregate docs usage. The site works without them.</p>
      <div class="cookie-consent__details">
        <h3>What is collected</h3>
        <p>When analytics is accepted, the site may collect aggregate documentation usage such as page views, navigation paths, and engagement events. The site should not send names, email addresses, account identifiers, or other personal content to analytics.</p>
        <h3>Your choices</h3>
        <p>You can accept or reject optional analytics cookies now and reopen these settings from the footer later.</p>
      </div>
    </div>
    <div class="cookie-consent__actions">
      <button type="button" class="button button--primary" data-cookie-accept>Accept analytics</button>
      <button type="button" class="button button--secondary" data-cookie-reject>Reject optional cookies</button>
    </div>
  `;

  banner.querySelector("[data-cookie-accept]").addEventListener("click", () => {
    applyConsent(setStoredConsent({analytics: true}));
    closeConsentUi();
  });
  banner.querySelector("[data-cookie-reject]").addEventListener("click", () => {
    applyConsent(setStoredConsent({analytics: false}));
    closeConsentUi();
  });

  document.body.appendChild(banner);
};

const bindPreferenceLinks = () => {
  document.querySelectorAll("[data-cookie-preferences]").forEach((button) => {
    if (button.dataset.cookieBound === "1") return;
    button.dataset.cookieBound = "1";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      renderPreferences();
    });
  });
};

const initConsent = () => {
  if (initialized) return;
  initialized = true;

  ensureGtag();
  window.neatDocsTrack = trackEvent;
  window.addEventListener(TRACK_EVENT, (event) => {
    const detail = event?.detail || {};
    trackEvent(detail.name, detail.params);
  });
  bindInteractionTracking();
  bindPreferenceLinks();

  const stored = getStoredConsent();
  if (stored) {
    applyConsent(stored);
    return;
  }

  const globalPrivacyControl = navigator.globalPrivacyControl === true;
  if (globalPrivacyControl) {
    updateGtagConsent({analytics: false});
  }
  renderBanner();
};

export function onClientEntry() {
  if (!ExecutionEnvironment.canUseDOM) return;
  initConsent();
}

export function onRouteDidUpdate() {
  if (!ExecutionEnvironment.canUseDOM) return;

  // onClientEntry does not fire reliably in the deployed build, so run the
  // one-time init here too (it is idempotent). Without this, gtag is never
  // bootstrapped on a page load and trackPageView() silently no-ops, so the
  // only page_view ever recorded is the one emitted by the consent banner's
  // accept handler — which is why analytics only captured the landing page.
  initConsent();

  bindPreferenceLinks();
  const stored = getStoredConsent();
  if (!stored) {
    renderBanner();
    return;
  }
  if (stored.analytics) {
    // applyConsent is idempotent (loadGtag guards on gtagLoaded); it guarantees
    // gtag is loaded for returning visitors whose consent was stored on a
    // previous visit, so the view below is actually sent.
    applyConsent(stored);
    window.setTimeout(trackPageView, 80);
  }
}
