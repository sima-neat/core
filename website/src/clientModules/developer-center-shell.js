const DEFAULT_SHELL_BASE = "/";

function assetUrl(base, file) {
  const normalized = (base || DEFAULT_SHELL_BASE).replace(/\/?$/, "/");
  return `${normalized}${file}`;
}

function loadStylesheet(href) {
  if (document.querySelector(`link[data-developer-center-shell-css="${href}"]`)) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    link.dataset.developerCenterShellCss = href;
    link.onload = () => resolve();
    link.onerror = () => reject(new Error(`Unable to load ${href}`));
    document.head.appendChild(link);
  });
}

function loadScript(src) {
  if (window.DeveloperCenterShell) {
    return Promise.resolve();
  }
  if (document.querySelector(`script[data-developer-center-shell-js="${src}"]`)) {
    return new Promise((resolve) => {
      const timer = window.setInterval(() => {
        if (window.DeveloperCenterShell) {
          window.clearInterval(timer);
          resolve();
        }
      }, 25);
    });
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.developerCenterShellJs = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Unable to load ${src}`));
    document.head.appendChild(script);
  });
}

function ensureShellRoot() {
  let root = document.getElementById("developer-center-shell-root");
  if (root) return root;

  root = document.createElement("div");
  root.id = "developer-center-shell-root";
  document.body.prepend(root);
  return root;
}

async function mountShell() {
  const config = window.__NEAT_DEVELOPER_CENTER_SHELL__ || {};
  const base = config.base || DEFAULT_SHELL_BASE;

  try {
    await loadStylesheet(assetUrl(base, "developer-center-shell.css"));
    await loadScript(assetUrl(base, "developer-center-shell.js"));

    if (!window.DeveloperCenterShell) {
      throw new Error("DeveloperCenterShell global was not registered.");
    }

    document.documentElement.classList.add("developer-center-shell-enabled");
    await window.DeveloperCenterShell.mount(ensureShellRoot(), {active: "software"});
  } catch (err) {
    console.warn("Developer Center shell is unavailable; using the local software navbar.", err);
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountShell, {once: true});
  } else {
    mountShell();
  }
}
