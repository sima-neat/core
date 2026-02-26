const path = require("path");

const repo = process.env.GITHUB_REPOSITORY || "";
const repoParts = repo.split("/");
const org = process.env.DOCS_ORG || repoParts[0] || "manuel-roldan";
const project = process.env.DOCS_PROJECT || repoParts[1] || "PipelineSession";

const url = process.env.DOCS_URL || `https://${org}.github.io`;
const baseUrl = process.env.DOCS_BASE_URL || "/";
const algoliaAppId = process.env.DOCS_ALGOLIA_APP_ID || "REPLACE_ME";
const algoliaApiKey = process.env.DOCS_ALGOLIA_API_KEY || "REPLACE_ME";
const algoliaIndexName = process.env.DOCS_ALGOLIA_INDEX_NAME || "REPLACE_ME";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "SiMa NEAT",
  tagline: "SiMa NEAT documentation",
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
      title: "SiMa NEAT",
      items: [
        { type: "doc", docId: "index", label: "Docs", position: "left" },
        { label: "C++ API", to: "/reference/cppapi/", position: "left" },
        { label: "Python API", to: "/reference/pythonapi/", position: "left" },
        { type: "search", position: "right" },
        {
          type: "html",
          position: "right",
          value:
            '<div class="language-pref"><label for="language-pref-select">Language</label><select id="language-pref-select" aria-label="Preferred language"><option value="cpp">C++</option><option value="py">Python</option></select></div>',
        },
        {
          href: `https://github.com/${org}/${project}`,
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        { label: "SiMa.ai NEAT Framework Documentation", to: "/" },
      ],
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
  },
  clientModules: [require.resolve("./src/clientModules/language-preference.js")],
};

module.exports = config;
