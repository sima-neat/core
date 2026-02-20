(() => {
  const PREF_KEY = "neat-docs-language";

  const getPreferredLang = () => {
    const stored = window.localStorage.getItem(PREF_KEY);
    return stored || "cpp";
  };

  const setPreferredLang = (lang) => {
    window.localStorage.setItem(PREF_KEY, lang);
  };

  const activateGroup = (tabGroup, lang) => {
    const buttons = tabGroup.querySelectorAll(".code-tab-button");
    const panels = tabGroup.querySelectorAll(".code-tab-panel");
    const hasLang = Array.from(panels).some((panel) => panel.dataset.tab === lang);
    const selected =
      hasLang ? lang : (panels[0] && panels[0].dataset.tab) || lang;

    panels.forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.tab === selected);
    });
    buttons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.tab === selected);
    });
  };

  const detectLanguage = (node) => {
    if (!node || node.nodeType !== 1) return null;
    const classList = Array.from(node.classList || []);
    const classLang = classList.find((cls) => cls.startsWith("language-"));
    if (classLang) return normalizeLang(classLang.replace("language-", ""));
    const code = node.querySelector("code[class*='language-']");
    if (code) {
      const cls = Array.from(code.classList).find((c) => c.startsWith("language-"));
      if (cls) return normalizeLang(cls.replace("language-", ""));
    }
    return null;
  };

  const normalizeLang = (lang) => {
    if (!lang) return null;
    const lower = lang.toLowerCase();
    if (lower === "python") return "py";
    if (lower === "c++" || lower === "cc" || lower === "cxx") return "cpp";
    return lower;
  };

  const findEndMarker = (start) => {
    let node = start.nextElementSibling;
    while (node) {
      if (node.classList && node.classList.contains("code-tabs-end")) {
        return node;
      }
      node = node.nextElementSibling;
    }
    return null;
  };

  const buildTabs = (marker) => {
    if (marker.classList.contains("code-tabs--js")) return;
    const endMarker = findEndMarker(marker);
    if (!endMarker) return;

    const blocks = [];
    let node = marker.nextElementSibling;
    while (node && node !== endMarker) {
      const lang = detectLanguage(node);
      if (lang) {
        blocks.push({ node, lang });
      }
      node = node.nextElementSibling;
    }

    if (!blocks.length) return;

    const buttons = document.createElement("div");
    buttons.className = "code-tab-buttons";
    const panels = document.createDocumentFragment();
    const seen = new Set();

    blocks.forEach(({ node: block, lang }) => {
      if (!seen.has(lang)) {
        const btn = document.createElement("button");
        btn.className = "code-tab-button";
        btn.dataset.tab = lang;
        btn.type = "button";
        btn.textContent = lang === "py" ? "Python" : "C++";
        buttons.appendChild(btn);
        seen.add(lang);
      }

      const panel = document.createElement("div");
      panel.className = "code-tab-panel";
      panel.dataset.tab = lang;
      panel.appendChild(block);
      panels.appendChild(panel);
    });

    marker.appendChild(buttons);
    marker.appendChild(panels);
    marker.classList.add("code-tabs--js");
    endMarker.remove();
  };

  const initTabs = (root) => {
    const tabs = root.querySelectorAll(".code-tabs");
    const preferred = getPreferredLang();

    tabs.forEach((tabGroup) => {
      buildTabs(tabGroup);
      const defaultTab = tabGroup.dataset.default || "cpp";
      const initial = preferred || defaultTab;

      if (!tabGroup.classList.contains("code-tabs--js")) {
        return;
      }

      const buttons = tabGroup.querySelectorAll(".code-tab-button");
      buttons.forEach((button) => {
        button.addEventListener("click", () => {
          const lang = button.dataset.tab;
          setPreferredLang(lang);
          document.querySelectorAll(".code-tabs").forEach((group) =>
            activateGroup(group, lang)
          );
          syncLanguageSelect(lang);
        });
      });

      activateGroup(tabGroup, initial);
    });
  };

  const syncLanguageSelect = (lang) => {
    const select = document.getElementById("language-pref-select");
    if (!select) return;
    select.value = lang;
  };

  const initLanguageSelect = () => {
    const select = document.getElementById("language-pref-select");
    if (!select) return;
    const preferred = getPreferredLang();
    select.value = preferred;
    select.addEventListener("change", () => {
      const lang = select.value;
      setPreferredLang(lang);
      document.querySelectorAll(".code-tabs").forEach((group) => {
        activateGroup(group, lang);
      });
    });
  };

  const handleRouteChange = () => {
    initTabs(document);
    initLanguageSelect();
  };

  const patchHistory = () => {
    const push = history.pushState;
    history.pushState = function () {
      push.apply(this, arguments);
      window.dispatchEvent(new Event("neat:routechange"));
    };
    const replace = history.replaceState;
    history.replaceState = function () {
      replace.apply(this, arguments);
      window.dispatchEvent(new Event("neat:routechange"));
    };
  };

  const observeMutations = () => {
    const target = document.querySelector("main") || document.body;
    if (!target) return;
    const observer = new MutationObserver(() => {
      initTabs(document);
    });
    observer.observe(target, { childList: true, subtree: true });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      patchHistory();
      handleRouteChange();
      observeMutations();
    });
  } else {
    patchHistory();
    handleRouteChange();
    observeMutations();
  }

  window.addEventListener("popstate", handleRouteChange);
  window.addEventListener("neat:routechange", handleRouteChange);
})();
