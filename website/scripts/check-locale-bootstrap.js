"use strict";

// Runs the docs' inline locale redirect from docusaurus.config.js against a fake
// browser, so cookie and preference handling is checked without a build (#799).

const assert = require("node:assert/strict");
const path = require("node:path");
const vm = require("node:vm");

const configPath = path.resolve(__dirname, "../docusaurus.config.js");

function loadBootstrap(preferredLocale) {
  const saved = process.env.DOCS_PREFERRED_LOCALE;
  if (preferredLocale === undefined) {
    delete process.env.DOCS_PREFERRED_LOCALE;
  } else {
    process.env.DOCS_PREFERRED_LOCALE = preferredLocale;
  }
  try {
    delete require.cache[configPath];
    const config = require(configPath);
    const tag = config.headTags.find(
      (entry) => entry.tagName === "script" && (entry.innerHTML || "").includes("sima-neat-locale"),
    );
    assert.ok(tag, "locale bootstrap script not found in headTags");
    return {source: tag.innerHTML, baseUrl: config.baseUrl};
  } finally {
    if (saved === undefined) {
      delete process.env.DOCS_PREFERRED_LOCALE;
    } else {
      process.env.DOCS_PREFERRED_LOCALE = saved;
    }
  }
}

function run({cookie = "", pathname = "/", developerCenterLocale, preferredLocale} = {}) {
  const {source, baseUrl} = loadBootstrap(preferredLocale);
  const redirects = [];
  const window = {
    location: {
      pathname: baseUrl + pathname.replace(/^\//, ""),
      search: "",
      hash: "",
      replace: (target) => redirects.push(target),
    },
    sessionStorage: {getItem: () => null, setItem: () => {}},
  };
  if (developerCenterLocale) {
    window.__NEAT_DOCS_PREFERRED_LOCALE__ = developerCenterLocale;
  }
  vm.runInNewContext(source, {window, document: {cookie}, URLSearchParams});
  return {redirects, baseUrl};
}

const base = (target, baseUrl) => baseUrl + target;

// A valid cookie still selects its locale, encoded or not.
{
  const {redirects, baseUrl} = run({cookie: "sima-neat-locale=ko"});
  assert.deepEqual(redirects, [base("ko/", baseUrl)]);
}
{
  const {redirects, baseUrl} = run({cookie: "other=1; sima-neat-locale=zh%2DHant", pathname: "/tutorials/"});
  assert.deepEqual(redirects, [base("zh-Hant/tutorials/", baseUrl)]);
}

// A malformed cookie no longer throws, and leaves the English page alone.
for (const value of ["%", "%E0%A4%A", "ja%"]) {
  assert.doesNotThrow(() => run({cookie: `sima-neat-locale=${value}`}), `cookie ${value}`);
  assert.deepEqual(run({cookie: `sima-neat-locale=${value}`}).redirects, [], `cookie ${value}`);
}

// With a malformed cookie, the Developer Center and build-time preferences still apply.
{
  const {redirects, baseUrl} = run({cookie: "sima-neat-locale=%", developerCenterLocale: "ja"});
  assert.deepEqual(redirects, [base("ja/", baseUrl)]);
}
{
  const {redirects, baseUrl} = run({cookie: "sima-neat-locale=%", preferredLocale: "uk", pathname: "/getting-started/"});
  assert.deepEqual(redirects, [base("uk/getting-started/", baseUrl)]);
}

console.log("Locale bootstrap checks passed.");
