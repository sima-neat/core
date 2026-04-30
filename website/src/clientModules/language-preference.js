import ExecutionEnvironment from "@docusaurus/ExecutionEnvironment";
import {trackDocsEvent} from "@site/src/lib/analytics";

const PREF_KEY = "neat-docs-language";

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
  const select = document.getElementById("language-pref-select");
  if (!select) return;
  select.value = normalizeLang(lang);
};

const initLanguageSelect = () => {
  const select = document.getElementById("language-pref-select");
  if (!select || select.dataset.bound === "1") return;
  select.dataset.bound = "1";
  syncLanguageSelect(getPreferredLang());

  select.addEventListener("change", () => {
    const lang = normalizeLang(select.value);
    setPreferredLang(lang);
    trackDocsEvent("language_tab_select", {
      language: lang,
      source: "navbar",
    });
  });

  window.addEventListener("neat:langchange", (event) => {
    const next = event?.detail?.lang || getPreferredLang();
    syncLanguageSelect(next);
  });
};

const scheduleInit = () => {
  window.requestAnimationFrame(() => {
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
