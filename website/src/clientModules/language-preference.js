import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";
import {trackDocsEvent} from "@site/src/lib/analytics";

const PREF_KEY = "neat-docs-language";
const LANGUAGE_AWARE_SELECTOR = "[data-neat-language-aware='true']";
let visibilityObserver;

const normalizeLang = (lang) => {
  if (!lang) return "cpp";
  const lower = String(lang).toLowerCase();
  if (lower === "python") return "py";
  if (lower === "c++" || lower === "cc" || lower === "cxx") return "cpp";
  return lower;
};

const getPreferredLang = () => {
  const stored = window.localStorage.getItem(PREF_KEY);
  return normalizeLang(stored || "cpp");
};

const setPreferredLang = (lang) => {
  const normalized = normalizeLang(lang);
  window.localStorage.setItem(PREF_KEY, normalized);
  window.dispatchEvent(new CustomEvent("neat:langchange", {detail: {lang: normalized}}));
};

const syncLanguageSelect = (lang) => {
  document.querySelectorAll("[data-language-pref-select]").forEach((select) => {
    select.value = normalizeLang(lang);
  });
};

const syncLanguageSelectorVisibility = () => {
  const languageAwarePage = Boolean(document.querySelector(LANGUAGE_AWARE_SELECTOR));
  document.documentElement.classList.toggle("neat-docs-language-aware", languageAwarePage);
};

const scheduleLanguageSelectorVisibilitySync = () => {
  window.requestAnimationFrame(syncLanguageSelectorVisibility);
};

const observeLanguageAwareContent = () => {
  if (visibilityObserver || !document.body) return;
  visibilityObserver = new MutationObserver(scheduleLanguageSelectorVisibilitySync);
  visibilityObserver.observe(document.body, {
    childList: true,
    subtree: true,
  });
};

const initLanguageSelect = () => {
  const selects = document.querySelectorAll("[data-language-pref-select]");
  if (!selects.length) return;

  selects.forEach((select) => {
    if (select.dataset.bound === "1") return;
    select.dataset.bound = "1";

    select.addEventListener("change", () => {
      const lang = normalizeLang(select.value);
      setPreferredLang(lang);
      trackDocsEvent("language_tab_select", {
        language: lang,
        source: "navbar",
      });
    });
  });

  syncLanguageSelect(getPreferredLang());

  window.addEventListener("neat:langchange", (event) => {
    const next = event?.detail?.lang || getPreferredLang();
    syncLanguageSelect(next);
  });
};

const scheduleInit = () => {
  window.requestAnimationFrame(() => {
    observeLanguageAwareContent();
    syncLanguageSelectorVisibility();
    initLanguageSelect();
    syncLanguageSelect(getPreferredLang());
  });
};

export function onClientEntry() {
  if (!ExecutionEnvironment.canUseDOM) return;
  scheduleInit();
}

export function onRouteDidUpdate() {
  if (!ExecutionEnvironment.canUseDOM) return;
  scheduleInit();
}
