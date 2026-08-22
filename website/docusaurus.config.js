const fs = require("fs");
const path = require("path");
const {execSync} = require("child_process");

function canonicalDocsPath(rawPath) {
  // Docusaurus' MDX-loader rule matches files against `contentDirs` after
  // webpack realpath-resolves them. When DOCS_PATH is an absolute path that
  // traverses a symlink (common when build.sh is invoked from a symlinked
  // checkout), the rule's include path keeps the symlink while loaded files
  // arrive with their realpath, so the rule never matches and pages render
  // without metadata. Canonicalize through realpath so both sides agree.
  const resolved = path.resolve(__dirname, rawPath);
  try {
    return fs.realpathSync(resolved);
  } catch {
    return resolved;
  }
}

function gitValue(command) {
  try {
    return execSync(command, {
      cwd: __dirname,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return "";
  }
}

function parseGitHubRepo(remoteUrl) {
  const match = (remoteUrl || "").match(/github\.com[:/]([^/]+)\/(.+?)(?:\.git)?$/);

  if (!match) {
    return null;
  }

  return {
    org: match[1],
    project: match[2],
  };
}

function normalizeBranch(branch) {
  return (branch || "")
    .replace(/^refs\/heads\//, "")
    .replace(/^origin\//, "")
    .trim();
}

const remoteRepo = parseGitHubRepo(gitValue("git config --get remote.origin.url"));
const repo =
  process.env.GITHUB_REPOSITORY ||
  (remoteRepo ? `${remoteRepo.org}/${remoteRepo.project}` : "");
const repoParts = repo.split("/");
const org = process.env.DOCS_ORG || repoParts[0] || "sima-neat";
const project = process.env.DOCS_PROJECT || repoParts[1] || "core";
const githubRepoUrl =
  process.env.DOCS_REPO_URL || `https://github.com/${org}/${project}`;
const githubOrgUrl = process.env.DOCS_GITHUB_ORG_URL || `https://github.com/${org}`;

const url = process.env.DOCS_URL || `https://${org}.github.io`;
const baseUrl = process.env.DOCS_BASE_URL || "/";
const docsLocales = ["en", "ko", "ja", "zh-Hant", "uk"];
const configuredPreferredLocale = docsLocales.includes(process.env.DOCS_PREFERRED_LOCALE)
  ? process.env.DOCS_PREFERRED_LOCALE
  : "";
const insightOpenApiSpec = path.resolve(
  __dirname,
  process.env.INSIGHT_OPENAPI_SPEC || "../../insight/neat_insight/openapi.json",
);
const insightApiOutput = path.resolve(
  __dirname,
  process.env.INSIGHT_API_OUTPUT || "../docs/tools/insight/api",
);
const siteRoot = url.replace(/\/+$/, "");
const developerCenterShellBase = process.env.DOCS_DEVELOPER_CENTER_SHELL_BASE || "";
const analyticsConfig = {
  measurementId: process.env.DOCS_GA_MEASUREMENT_ID || "",
};
const footerLinks = [
  { label: "SiMa.ai Neat Framework Documentation", to: "/" },
  { html: '<button type="button" class="footer__link-item cookie-preferences-link" data-cookie-preferences>Cookie preferences</button>' },
];

const buildBranch = normalizeBranch(
  process.env.DOCS_BUILD_BRANCH ||
    process.env.GITHUB_HEAD_REF ||
    process.env.GITHUB_REF_NAME ||
    gitValue("git rev-parse --abbrev-ref HEAD"),
);
const buildCommit = (
  process.env.DOCS_BUILD_COMMIT ||
  process.env.GITHUB_SHA ||
  gitValue("git rev-parse HEAD")
).trim();
const buildTime = (
  process.env.DOCS_BUILD_TIME || new Date().toISOString()
).trim();
const showBuildBanner = Boolean(buildBranch && buildBranch !== "main");
const buildBranchUrl = buildBranch
  ? `${githubRepoUrl}/tree/${encodeURI(buildBranch)}`
  : "";
const buildCommitUrl = buildCommit ? `${githubRepoUrl}/commit/${buildCommit}` : "";

/** @type {import('@docusaurus/types').Config} */
const config = {
  i18n: {
    defaultLocale: "en",
    locales: docsLocales,
    localeConfigs: {
      en: {label: "English", htmlLang: "en-US"},
      ko: {label: "한국어", htmlLang: "ko-KR"},
      ja: {label: "日本語", htmlLang: "ja-JP"},
      "zh-Hant": {label: "繁體中文", htmlLang: "zh-Hant-TW"},
      uk: {label: "Українська", htmlLang: "uk-UA"},
    },
  },
  future: {
    faster: {
      rspackBundler: true,
      swcJsLoader: true,
    },
  },
  title: "SiMa.ai Neat",
  tagline: "SiMa.ai Neat documentation",
  url,
  baseUrl,
  onBrokenLinks: "throw",
  markdown: {
    format: "md",
    hooks: {
      onBrokenMarkdownLinks: "throw",
    },
  },
  favicon: "img/favicon.png",
  staticDirectories: ["static", "../docs/images", "../docs/develop-apps/images"],
  organizationName: org,
  projectName: project,
  headTags: [
    {
      tagName: "script",
      attributes: {},
      innerHTML: `(() => {
        const supported = ${JSON.stringify(docsLocales)};
        const localized = supported.filter((locale) => locale !== "en");
        window.__NEAT_DOCS_PREFERRED_LOCALE__ =
          window.__NEAT_DOCS_PREFERRED_LOCALE__ || ${JSON.stringify(configuredPreferredLocale)};
        const params = new URLSearchParams(window.location.search);
        const previewKey = "neat-docs-i18n-preview";
        let previewEnabled = params.get("i18n") === "1";
        try {
          if (previewEnabled) {
            window.sessionStorage.setItem(previewKey, "1");
          } else {
            previewEnabled = window.sessionStorage.getItem(previewKey) === "1";
          }
        } catch {}
        if (previewEnabled) return;
        const base = ${JSON.stringify(baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`)};
        const cookie = document.cookie.match(/(?:^|; )sima-neat-locale=([^;]*)/);
        const cookieLocale = cookie ? decodeURIComponent(cookie[1]) : "";
        const preferred = supported.includes(window.__NEAT_DOCS_PREFERRED_LOCALE__)
          ? window.__NEAT_DOCS_PREFERRED_LOCALE__
          : (supported.includes(cookieLocale) ? cookieLocale : "en");
        let current = "en";
        let suffix = window.location.pathname.startsWith(base)
          ? window.location.pathname.slice(base.length)
          : window.location.pathname.replace(/^[/]/, "");
        for (const locale of localized) {
          if (suffix === locale || suffix.startsWith(locale + "/")) {
            current = locale;
            suffix = suffix === locale ? "" : suffix.slice(locale.length + 1);
            break;
          }
        }
        if (current === preferred) return;
        const localizedPath = preferred === "en"
          ? base + suffix
          : base + preferred + "/" + suffix;
        window.location.replace(localizedPath + window.location.search + window.location.hash);
      })();`,
    },
    {
      tagName: "link",
      attributes: {
        rel: "preconnect",
        href: "https://fonts.googleapis.com",
      },
    },
    {
      tagName: "link",
      attributes: {
        rel: "preconnect",
        href: "https://fonts.gstatic.com",
        crossorigin: "anonymous",
      },
    },
    {
      tagName: "link",
      attributes: {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;800&display=swap",
      },
    },
    {
      tagName: "script",
      attributes: {},
      innerHTML: `window.__NEAT_DEVELOPER_CENTER_SHELL__ = ${JSON.stringify({
        enabled: Boolean(developerCenterShellBase),
        base: developerCenterShellBase,
      })};`,
    },
    {
      tagName: "script",
      attributes: {},
      innerHTML: `window.__NEAT_DOCS_ANALYTICS__ = ${JSON.stringify(analyticsConfig)};`,
    },
  ],
  presets: [
    [
      "classic",
      {
        docs: {
          path: canonicalDocsPath(process.env.DOCS_PATH || "../docs"),
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
          sidebarItemsGenerator: require("./sidebarItemsGenerator.js"),
          docItemComponent: "@theme/ApiItem",
          exclude: ["doxygen/**", "reference/env_var_rationalization.md"],
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
  plugins: [
    [
      "@docusaurus/plugin-client-redirects",
      {
        redirects: [
          {
            from: "/getting-started/dev-environment/pair-with-a-devkit/",
            to: "/getting-started/dev-environment/devkit-sync/",
          },
          {
            from: "/getting-started/dev-environment/run-on-the-devkit/",
            to: "/getting-started/dev-environment/devkit-sync/",
          },
          {
            from: "/tutorials/beginner/",
            to: "/tutorials/",
          },
          {
            from: "/tutorials/intermediate/",
            to: "/tutorials/",
          },
          {
            from: "/tutorials/advanced/",
            to: "/tutorials/",
          },
          {
            from: "/tutorials/measure-pcie-detection-throughput/",
            to: "/tutorials/run-pcie-inference-async/",
          },
          {
            from: "/tutorials/run-pcie-inference-modes/",
            to: "/tutorials/run-your-first-model-over-pcie/",
          },
          {
            from: "/tutorials/run-models-on-multiple-pcie-queues/",
            to: "/tutorials/run-multiple-models/",
          },
        ],
      },
    ],
    [
      "docusaurus-plugin-openapi-docs",
      {
        id: "openapi",
        docsPluginId: "classic",
        config: {
          insight: {
            specPath: insightOpenApiSpec,
            outputDir: insightApiOutput,
            template: path.resolve(
              __dirname,
              "openapi-templates/api.mdx.mustache",
            ),
            tagTemplate: path.resolve(
              __dirname,
              "openapi-templates/tag.mdx.mustache",
            ),
            downloadUrl:
              "https://raw.githubusercontent.com/sima-neat/insight/main/neat_insight/openapi.json",
            hideSendButton: true,
            sidebarOptions: {
              groupPathsBy: "tagGroup",
              categoryLinkSource: "tag",
            },
          },
        },
      },
    ],
  ],
  themes: ["docusaurus-theme-openapi-docs"],
  themeConfig: {
    docs: {
      sidebar: {
        autoCollapseCategories: true,
      },
    },
    navbar: {
      title: "SiMa.ai Neat",
      items: [
        { label: "Getting Started", to: "/getting-started/", position: "left" },
        { label: "C++ API", to: "/reference/cppapi/", position: "left" },
        { label: "Python API", to: "/reference/pythonapi/", position: "left" },
        {
          type: "html",
          position: "left",
          value:
            '<div class="language-pref"><label for="language-pref-select">Code</label><select id="language-pref-select" data-language-pref-select aria-label="Preferred programming language"><option value="cpp">C++</option><option value="py">Python</option></select></div>',
        },
      ],
    },
    colorMode: {
      disableSwitch: true,
    },
    footer: {
      style: "dark",
      links: footerLinks,
      copyright: `Copyright © ${new Date().getFullYear()} SiMa.ai`,
    },
  },
  customFields: {
    buildInfo: {
      showBanner: showBuildBanner,
      branch: buildBranch,
      branchUrl: buildBranchUrl,
      commit: buildCommit.slice(0, 12),
      commitUrl: buildCommitUrl,
      builtAt: buildTime.replace("T", " ").replace(/\.\d{3}Z$/, " UTC"),
    },
    githubRepoUrl,
    githubOrgUrl,
  },
  clientModules: [
    require.resolve("./src/clientModules/analytics-consent.js"),
    require.resolve("./src/clientModules/developer-center-shell.js"),
    require.resolve("./src/clientModules/developer-center-nav.js"),
    require.resolve("./src/clientModules/global-theme.js"),
    require.resolve("./src/clientModules/language-preference.js"),
    require.resolve("./src/clientModules/search-highlight.js"),
    require.resolve("./src/clientModules/collapse-sidebar-on-home.js"),
    require.resolve("./src/clientModules/strip-category-ssr-href.js"),
    require.resolve("./src/clientModules/i18n-preview.js"),
  ],
};

module.exports = config;
