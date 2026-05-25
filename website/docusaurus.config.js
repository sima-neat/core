const {execSync} = require("child_process");

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

const url = process.env.DOCS_URL || `https://${org}.github.io`;
const baseUrl = process.env.DOCS_BASE_URL || "/";
const siteRoot = url.replace(/\/+$/, "");
const developerCenterRoot = `${siteRoot}/`;
const hardwareUrl = `${siteRoot}/docs/hardware`;
const examplesUrl = `${siteRoot}/examples/`;
const algoliaAppId = process.env.DOCS_ALGOLIA_APP_ID || "REPLACE_ME";
const algoliaApiKey = process.env.DOCS_ALGOLIA_API_KEY || "REPLACE_ME";
const algoliaIndexName = process.env.DOCS_ALGOLIA_INDEX_NAME || "REPLACE_ME";
const docsGaMeasurementId = process.env.DOCS_GA_MEASUREMENT_ID || "";
const docsAnalyticsConfig = {
  measurementId: docsGaMeasurementId,
};
const footerLinks = [
  { label: "SiMa.ai Neat Framework Documentation", to: "/" },
  {
    html: '<button type="button" class="footer__link-item cookie-preferences-link" data-cookie-preferences>Cookie preferences</button>',
  },
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
  title: "SiMa Neat",
  tagline: "SiMa Neat documentation",
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
  organizationName: org,
  projectName: project,
  headTags: [
    {
      tagName: "script",
      attributes: {},
      innerHTML: `window.__NEAT_DOCS_ANALYTICS__ = ${JSON.stringify(
        docsAnalyticsConfig,
      )};`,
    },
  ],
  presets: [
    [
      "classic",
      {
        docs: {
          path: process.env.DOCS_PATH || "../docs",
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
          exclude: ["doxygen/**", "_tmp_test.txt"],
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
  plugins: [],
  themeConfig: {
    navbar: {
      title: "Developer Center",
      logo: {
        alt: "SiMa.ai",
        src: "img/favicon.png",
        href: developerCenterRoot,
      },
      items: [
        {
          type: "html",
          position: "left",
          value: `<a class="navbar__item navbar__link" href="${hardwareUrl}">Hardware</a>`,
        },
        { to: "/", label: "Software", position: "left" },
        {
          type: "html",
          position: "left",
          value: `<a class="navbar__item navbar__link" href="${examplesUrl}">Examples</a>`,
        },
        { href: "https://huggingface.co/simaai", label: "Models", position: "left" },
        { href: "https://developer.sima.ai", label: "Community", position: "left" },
        { type: "doc", docId: "index", label: "Docs", position: "right" },
        { label: "C++ API", to: "/reference/cppapi/", position: "right" },
        { label: "Python API", to: "/reference/pythonapi/", position: "right" },
        { type: "search", position: "right" },
        {
          type: "html",
          position: "right",
          value:
            '<div class="language-pref"><label for="language-pref-select">Language</label><select id="language-pref-select" aria-label="Preferred language"><option value="cpp">C++</option><option value="py">Python</option></select></div>',
        },
        {
          href: githubRepoUrl,
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: footerLinks,
      copyright: `Copyright © ${new Date().getFullYear()} SiMa.ai`,
    },
    algolia: {
      appId: algoliaAppId,
      apiKey: algoliaApiKey,
      indexName: algoliaIndexName,
    },
  },
  customFields: {
    algoliaSearch: {
      appId: algoliaAppId,
      apiKey: algoliaApiKey,
      indexName: algoliaIndexName,
    },
    buildInfo: {
      showBanner: showBuildBanner,
      branch: buildBranch,
      branchUrl: buildBranchUrl,
      commit: buildCommit.slice(0, 12),
      commitUrl: buildCommitUrl,
      builtAt: buildTime.replace("T", " ").replace(/\.\d{3}Z$/, " UTC"),
    },
    analytics: docsAnalyticsConfig,
  },
  clientModules: [
    require.resolve("./src/clientModules/analytics-consent.js"),
    require.resolve("./src/clientModules/developer-center-nav.js"),
    require.resolve("./src/clientModules/language-preference.js"),
    require.resolve("./src/clientModules/search-highlight.js"),
  ],
};

module.exports = config;
