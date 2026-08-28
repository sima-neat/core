const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const docusaurusConfig = require("../docusaurus.config.js");

for (const tag of docusaurusConfig.headTags || []) {
  if (tag.tagName === "script" && typeof tag.innerHTML === "string") {
    assert.doesNotThrow(() => new vm.Script(tag.innerHTML));
  }
}

const clientModule = fs.readFileSync(
  path.resolve(__dirname, "../src/clientModules/i18n-preview.js"),
  "utf8",
);
const shellHostModule = fs.readFileSync(
  path.resolve(__dirname, "../src/clientModules/developer-center-shell.js"),
  "utf8",
);
const rootTheme = fs.readFileSync(
  path.resolve(__dirname, "../src/theme/Root.js"),
  "utf8",
);

const rootPreferenceLogic = rootTheme.match(
  /function preferredDocsLocale\(\) \{[\s\S]*?^\}\n\nfunction localizationEnabled\(search, currentLocale\) \{[\s\S]*?^\}/m,
);
assert.ok(rootPreferenceLogic, "Root locale preference helpers must remain testable");

const blockedStorageContext = {
  I18N_PREVIEW_PARAM: "i18n",
  I18N_PREVIEW_SESSION_KEY: "neat-docs-i18n-preview",
  DOCS_LOCALE_COOKIE: "sima-neat-locale",
  URLSearchParams,
  decodeURIComponent,
  document: {cookie: "sima-neat-locale=ja"},
  window: {
    __NEAT_DOCS_PREFERRED_LOCALE__: "",
    sessionStorage: {
      getItem() {
        throw new Error("session storage blocked");
      },
    },
  },
};
vm.runInNewContext(
  `${rootPreferenceLogic[0]}; localeEnabled = localizationEnabled("", "ja");`,
  blockedStorageContext,
);
assert.equal(
  blockedStorageContext.localeEnabled,
  true,
  "A blocked session store must not suppress the configured locale cookie",
);

const malformedCookieContext = {
  I18N_PREVIEW_PARAM: "i18n",
  I18N_PREVIEW_SESSION_KEY: "neat-docs-i18n-preview",
  DOCS_LOCALE_COOKIE: "sima-neat-locale",
  URLSearchParams,
  decodeURIComponent,
  document: {cookie: "sima-neat-locale=%"},
  window: {
    __NEAT_DOCS_PREFERRED_LOCALE__: "",
    sessionStorage: {getItem: () => null},
  },
};
assert.doesNotThrow(() => {
  vm.runInNewContext(
    `${rootPreferenceLogic[0]}; localeEnabled = localizationEnabled("", "ja");`,
    malformedCookieContext,
  );
});
assert.equal(
  malformedCookieContext.localeEnabled,
  false,
  "A malformed locale cookie must be ignored",
);

const tutorialAvailabilityLogic = rootTheme.match(
  /function tutorialAvailableInLocale\([\s\S]*?^\}/m,
);
assert.ok(tutorialAvailabilityLogic, "Tutorial locale availability must remain testable");

const tutorialAvailabilityContext = {};
vm.runInNewContext(
  `${tutorialAvailabilityLogic[0]}; tutorialAvailable = tutorialAvailableInLocale;`,
  tutorialAvailabilityContext,
);
assert.equal(
  tutorialAvailabilityContext.tutorialAvailable(
    "/develop-apps/tutorials/001_run_your_first_model/",
    "en",
    "en",
    ["ko", "ja", "zh-Hant", "uk"],
  ),
  true,
  "Tutorials in the default English locale must not show the English-only banner",
);
assert.equal(
  tutorialAvailabilityContext.tutorialAvailable(
    "/develop-apps/tutorials/001_run_your_first_model/",
    "ja",
    "en",
    ["ko", "ja", "zh-Hant", "uk"],
  ),
  true,
  "Tutorials translated into the current locale must not show the English-only banner",
);
assert.equal(
  tutorialAvailabilityContext.tutorialAvailable(
    "/develop-apps/tutorials/001_run_your_first_model/",
    "fr",
    "en",
    ["ko", "ja", "zh-Hant", "uk"],
  ),
  false,
  "Tutorials without a translation must retain the English-only banner",
);

assert.match(
  shellHostModule,
  /export function onRouteDidUpdate\(\)[\s\S]*requestAnimationFrame[\s\S]*mountShell\(\)/,
  "Developer Center shell must remount after client-side locale route changes",
);

for (const localizedAutodocRoute of [
  "/compile-a-model",
  "/genai-llima",
  "/tools/insight",
  "/tools/sentinel",
  "/tools/sima-cli",
]) {
  assert.doesNotMatch(
    rootTheme,
    new RegExp(`^[ \\t]*["']${localizedAutodocRoute.replaceAll("/", "\\/")}["'],?$`, "m"),
    `${localizedAutodocRoute} must not show the English-only localization banner`,
  );
}

assert.match(
  rootTheme,
  /ENGLISH_ONLY_ROUTE_PREFIXES\s*=\s*\[[\s\S]*["']\/reference\/cppapi["'][\s\S]*["']\/reference\/pythonapi["'][\s\S]*\]/,
  "Generated C++ and Python API routes must show the English-only localization banner",
);

function anchor(href, attributes = {}) {
  return {
    attributes: {href, ...attributes},
    getAttribute(name) {
      return this.attributes[name] ?? null;
    },
    hasAttribute(name) {
      return Object.hasOwn(this.attributes, name);
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
  };
}

function runModule({
  href,
  pathname,
  search,
  anchors = [],
  cookie = "",
  preferredLocale = "",
  previewSession = false,
}) {
  let clickHandler;
  let assigned;
  const session = new Map(previewSession ? [["neat-docs-i18n-preview", "1"]] : []);
  const context = {
    URL,
    URLSearchParams,
    window: {
      location: {
        assign(destination) {
          assigned = destination;
        },
        href,
        origin: new URL(href).origin,
        pathname,
        search,
      },
      __NEAT_DOCS_PREFERRED_LOCALE__: preferredLocale,
      sessionStorage: {
        getItem(key) {
          return session.get(key) ?? null;
        },
        setItem(key, value) {
          session.set(key, value);
        },
      },
    },
    document: {
      cookie,
      documentElement: {},
      addEventListener(type, handler) {
        if (type === "click") clickHandler = handler;
      },
      querySelectorAll() {
        return anchors;
      },
    },
    MutationObserver: class {
      observe() {}
    },
  };
  vm.runInNewContext(clientModule, context);
  return {
    click(anchorValue, eventOverrides = {}) {
      let defaultPrevented = false;
      let propagationStopped = false;
      let immediatePropagationStopped = false;
      clickHandler({
        altKey: false,
        button: 0,
        ctrlKey: false,
        metaKey: false,
        shiftKey: false,
        preventDefault() {
          defaultPrevented = true;
        },
        stopImmediatePropagation() {
          immediatePropagationStopped = true;
        },
        stopPropagation() {
          propagationStopped = true;
        },
        target: {closest: () => anchorValue},
        ...eventOverrides,
      });
      return {defaultPrevented, immediatePropagationStopped, propagationStopped};
    },
    get assigned() {
      return assigned;
    },
    session,
  };
}

const localized = anchor("/develop-apps/hello-neat/minimal/");
const alreadyLocalized = anchor("/zh-Hant/develop-apps/hello-neat/run_first_model/");
const external = anchor("https://example.com/docs");
const hash = anchor("#section");
const preview = runModule({
  anchors: [localized, alreadyLocalized, external, hash],
  href: "http://localhost:3100/zh-Hant/getting-started/?i18n=1",
  pathname: "/zh-Hant/getting-started/",
  search: "?i18n=1",
});

assert.equal(
  localized.getAttribute("href"),
  "/zh-Hant/develop-apps/hello-neat/minimal/?i18n=1",
);
assert.equal(
  alreadyLocalized.getAttribute("href"),
  "/zh-Hant/develop-apps/hello-neat/run_first_model/?i18n=1",
);
assert.equal(external.getAttribute("href"), "https://example.com/docs");
assert.equal(hash.getAttribute("href"), "#section");

const dynamic = anchor("/develop-apps/hello-neat/minimal/");
const clickResult = preview.click(dynamic);
assert.equal(preview.assigned, "/zh-Hant/develop-apps/hello-neat/minimal/?i18n=1");
assert.deepEqual(clickResult, {
  defaultPrevented: true,
  immediatePropagationStopped: true,
  propagationStopped: true,
});

const modifiedPreview = runModule({
  href: "http://localhost:3100/ja/getting-started/?i18n=1",
  pathname: "/ja/getting-started/",
  search: "?i18n=1",
});
modifiedPreview.click(anchor("/develop-apps/hello-neat/minimal/"), {metaKey: true});
assert.equal(modifiedPreview.assigned, undefined);

const basePathLink = anchor("/docs/develop-apps/hello-neat/minimal/");
const sharedImage = anchor("/docs/images/tutorial_sample_image.png");
const relativeSharedImage = anchor("../../images/tutorial_sample_image.png");
const explicitDownload = anchor("/docs/downloads/model", {download: ""});
runModule({
  anchors: [basePathLink, sharedImage, relativeSharedImage, explicitDownload],
  href: "http://localhost:3100/docs/ko/getting-started/?i18n=1",
  pathname: "/docs/ko/getting-started/",
  search: "?i18n=1",
});
assert.equal(
  basePathLink.getAttribute("href"),
  "/docs/ko/develop-apps/hello-neat/minimal/?i18n=1",
);
assert.equal(sharedImage.getAttribute("href"), "/docs/images/tutorial_sample_image.png");
assert.equal(relativeSharedImage.getAttribute("href"), "../../images/tutorial_sample_image.png");
assert.equal(explicitDownload.getAttribute("href"), "/docs/downloads/model");

const staticAssetPreview = runModule({
  href: "http://localhost:3100/docs/ko/develop-apps/hello-neat/run_an_app/?i18n=1",
  pathname: "/docs/ko/develop-apps/hello-neat/run_an_app/",
  search: "?i18n=1",
});
const staticAssetClick = staticAssetPreview.click(
  anchor("/docs/images/tutorial_sample_image.png"),
);
assert.equal(staticAssetPreview.assigned, undefined);
assert.deepEqual(staticAssetClick, {
  defaultPrevented: false,
  immediatePropagationStopped: false,
  propagationStopped: false,
});

const disabled = anchor("/ja/getting-started/");
runModule({
  anchors: [disabled],
  href: "http://localhost:3100/ja/getting-started/",
  pathname: "/ja/getting-started/",
  search: "",
});
assert.equal(disabled.getAttribute("href"), "/ja/getting-started/");

assert.doesNotThrow(() =>
  runModule({
    cookie: "sima-neat-locale=%",
    href: "http://localhost:3100/ja/getting-started/",
    pathname: "/ja/getting-started/",
    search: "",
  }),
);

const ukrainian = anchor("/develop-apps/hello-neat/minimal/");
runModule({
  anchors: [ukrainian],
  href: "http://localhost:3100/uk/getting-started/?i18n=1",
  pathname: "/uk/getting-started/",
  search: "?i18n=1",
});
assert.equal(ukrainian.getAttribute("href"), "/uk/develop-apps/hello-neat/minimal/?i18n=1");

// Generated sidebar and pagination hrefs omit the preview query. A full page
// navigation must remain in preview mode using the current tab's session.
const sessionSidebar = anchor("/ja/getting-started/neat-library/");
const sessionPrevious = anchor("/ja/getting-started/dev-environment/");
const sessionNext = anchor("/ja/getting-started/compatibility");
runModule({
  anchors: [sessionSidebar, sessionPrevious, sessionNext],
  href: "http://localhost:3100/ja/getting-started/neat-library/",
  pathname: "/ja/getting-started/neat-library/",
  previewSession: true,
  search: "",
});
assert.equal(
  sessionSidebar.getAttribute("href"),
  "/ja/getting-started/neat-library/?i18n=1",
);
assert.equal(
  sessionPrevious.getAttribute("href"),
  "/ja/getting-started/dev-environment/?i18n=1",
);
assert.equal(
  sessionNext.getAttribute("href"),
  "/ja/getting-started/compatibility?i18n=1",
);

const queryPreview = runModule({
  href: "http://localhost:3100/ko/getting-started/?i18n=1",
  pathname: "/ko/getting-started/",
  search: "?i18n=1",
});
assert.equal(queryPreview.session.get("neat-docs-i18n-preview"), "1");

const globalPreferenceLink = anchor("/getting-started/neat-library/");
const globalPreference = runModule({
  anchors: [globalPreferenceLink],
  href: "http://localhost:3100/ja/getting-started/",
  pathname: "/ja/getting-started/",
  preferredLocale: "ja",
  search: "",
});
assert.equal(
  globalPreferenceLink.getAttribute("href"),
  "/ja/getting-started/neat-library/",
);
globalPreference.click(anchor("/getting-started/compatibility"));
assert.equal(globalPreference.assigned, "/ja/getting-started/compatibility");

console.log("Localized preview navigation checks passed.");
