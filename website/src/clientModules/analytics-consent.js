import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";

const CONSENT_KEY = "neat-docs-cookie-consent";
const CONSENT_VERSION = 1;
const ANALYTICS_EVENT = "neat:analytics-consent";

let gtagLoaded = false;
let initialPageViewSent = false;
let lastTrackedLocation = "";

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
  window.localStorage.setItem(CONSENT_KEY, JSON.stringify(next));
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
  const pagePath = `${window.location.pathname}${window.location.search}`;
  if (pagePath === lastTrackedLocation) return;
  lastTrackedLocation = pagePath;
  window.gtag("event", "page_view", {
    page_title: document.title,
    page_location: window.location.href,
    page_path: pagePath,
  });
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
  ensureGtag();
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
  bindPreferenceLinks();
  const stored = getStoredConsent();
  if (!stored) {
    renderBanner();
    return;
  }
  if (stored.analytics) {
    window.setTimeout(trackPageView, 80);
  }
}
