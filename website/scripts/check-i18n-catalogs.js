"use strict";

const fs = require("node:fs");
const path = require("node:path");

const websiteRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(websiteRoot, "..");
const locales = ["ko", "ja", "zh-Hant", "uk"];
const categoryPrefix = "sidebar.docs.category.";
const protectedLabels = new Set([
  "API", "GenAI", "Insight", "Model Compiler API", "Neat Library", "Neat SDK",
  "appzoo", "device", "mla", "modelzoo", "neat", "packages", "playbooks",
  "pyneat", "sdk", "sima-cli",
]);
const requiredLabels = new Set([
  "Advanced Concepts", "Application Design", "Build & Test", "Cameras & Streaming",
  "Compile a Model", "Contracts & Internals", "Contribute", "Data & Model Contracts",
  "Develop Apps", "Development Workflow", "Execution Model", "GenAI Model",
  "GenAI with LLiMa", "Getting Started", "Graphs & Pipelines", "Hello Neat!",
  "Model Runtime", "Models & Inference", "PCIe Co-Processing", "References",
  "Release & Maintenance", "Release Notes", "Start Here", "Tools", "Tutorials",
]);
const requiredShellCatalogKeys = {
  "code.json": [
    "theme.ErrorPageContent.title",
    "theme.NotFound.title",
    "theme.NotFound.p1",
    "theme.NotFound.p2",
    "theme.docs.paginator.previous",
    "theme.docs.paginator.next",
    "theme.navbar.mobileLanguageDropdown.label",
  ],
  "docusaurus-theme-classic/navbar.json": [
    "title",
    "item.label.Getting Started",
    "item.label.C++ API",
    "item.label.Python API",
  ],
  "docusaurus-theme-classic/footer.json": [
    "link.item.label.SiMa.ai Neat Framework Documentation",
    "copyright",
  ],
};
const failures = [];
const sidebarCatalogs = new Map();

const scope = JSON.parse(fs.readFileSync(path.join(websiteRoot, "i18n", "scope.json"), "utf8"));
const toolConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "sima-i18n.config.json"), "utf8"));
if (JSON.stringify(scope.excludedPrefixes) !== JSON.stringify(toolConfig.excludedPrefixes)) {
  failures.push("sima-i18n.config.json excludedPrefixes must match website/i18n/scope.json");
}

for (const locale of locales) {
  for (const [relativePath, requiredKeys] of Object.entries(requiredShellCatalogKeys)) {
    const shellCatalogPath = path.join(websiteRoot, "i18n", locale, relativePath);
    if (!fs.existsSync(shellCatalogPath)) {
      failures.push(`${locale} is missing its ${relativePath} UI catalog`);
      continue;
    }
    const shellCatalog = JSON.parse(fs.readFileSync(shellCatalogPath, "utf8"));
    for (const key of requiredKeys) {
      if (!shellCatalog[key]) failures.push(`${locale} ${relativePath} is missing key: ${key}`);
    }
  }
  const catalogPath = path.join(
    websiteRoot,
    "i18n",
    locale,
    "docusaurus-plugin-content-docs",
    "current.json",
  );
  if (!fs.existsSync(catalogPath)) {
    failures.push(`${locale} is missing its documentation sidebar catalog`);
    continue;
  }
  const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
  sidebarCatalogs.set(locale, catalog);
  for (const englishLabel of requiredLabels) {
    if (!catalog[`${categoryPrefix}${englishLabel}`]) {
      failures.push(`${locale} sidebar is missing category: ${englishLabel}`);
    }
  }
  for (const [id, translation] of Object.entries(catalog)) {
    if (!id.startsWith(categoryPrefix)) continue;
    const englishLabel = id.slice(categoryPrefix.length);
    if (translation.message === englishLabel && !protectedLabels.has(englishLabel)) {
      failures.push(`${locale} sidebar category remains in English: ${englishLabel}`);
    }
  }
}

const allSidebarKeys = new Set(
  [...sidebarCatalogs.values()].flatMap((catalog) => Object.keys(catalog)),
);
for (const [locale, catalog] of sidebarCatalogs) {
  for (const key of allSidebarKeys) {
    if (!catalog[key]) failures.push(`${locale} sidebar is missing generated key: ${key}`);
  }
}

if (failures.length > 0) {
  console.error("Localization catalog validation failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("Localized UI and sidebar catalogs are complete.");
