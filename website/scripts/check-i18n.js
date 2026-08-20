const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const {spawnSync} = require("child_process");

const websiteRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(websiteRoot, "..");
const locales = ["ko", "ja", "zh-Hant", "uk"];
const requireComplete = process.argv.includes("--require-complete");
const sourceHashes = JSON.parse(
  fs.readFileSync(path.join(websiteRoot, "i18n", "translation-sources.json"), "utf8"),
);
const localizationScope = JSON.parse(
  fs.readFileSync(path.join(websiteRoot, "i18n", "scope.json"), "utf8"),
);

const sidebarCategoryPrefix = "sidebar.docs.category.";
const protectedSidebarLabels = new Set([
  "API",
  "GenAI",
  "Insight",
  "Model Compiler API",
  "Neat Library",
  "Neat SDK",
  "appzoo",
  "device",
  "mla",
  "modelzoo",
  "neat",
  "packages",
  "playbooks",
  "pyneat",
  "sdk",
  "sima-cli",
]);
const requiredLocalizedSidebarLabels = new Set([
  "Advanced Concepts",
  "Application Design",
  "Build & Test",
  "Cameras & Streaming",
  "Compile a Model",
  "Contracts & Internals",
  "Contribute",
  "Data & Model Contracts",
  "Develop Apps",
  "Development Workflow",
  "Execution Model",
  "GenAI Model",
  "GenAI with LLiMa",
  "Getting Started",
  "Graphs & Pipelines",
  "Hello Neat!",
  "Model Runtime",
  "Models & Inference",
  "PCIe Co-Processing",
  "References",
  "Release & Maintenance",
  "Release Notes",
  "Start Here",
  "Tools",
  "Tutorials",
]);

const git = spawnSync(
  "git",
  ["ls-files", "-z", "--", "docs/**/*.md", "docs/**/*.mdx"],
  {cwd: repoRoot, encoding: "utf8"},
);

if (git.status !== 0) {
  console.error(git.stderr || "Unable to inventory documentation sources.");
  process.exit(1);
}

const allSourcePaths = git.stdout.split("\0").filter(Boolean).sort();
const excludedSourcePaths = allSourcePaths.filter((sourcePath) =>
  localizationScope.excludedPrefixes.some((prefix) => sourcePath.startsWith(prefix)),
);
const sourcePaths = allSourcePaths.filter((sourcePath) => !excludedSourcePaths.includes(sourcePath));
const failures = [];
const coverage = Object.fromEntries(locales.map((locale) => [locale, 0]));

for (const locale of locales) {
  const sidebarCatalogPath = path.join(
    websiteRoot,
    "i18n",
    locale,
    "docusaurus-plugin-content-docs",
    "current.json",
  );
  if (!fs.existsSync(sidebarCatalogPath)) {
    failures.push(`${locale} is missing its documentation sidebar catalog`);
    continue;
  }

  const sidebarCatalog = JSON.parse(fs.readFileSync(sidebarCatalogPath, "utf8"));
  for (const englishLabel of requiredLocalizedSidebarLabels) {
    const id = `${sidebarCategoryPrefix}${englishLabel}`;
    if (!sidebarCatalog[id]) {
      failures.push(`${locale} sidebar is missing category: ${englishLabel}`);
    }
  }
  for (const [id, translation] of Object.entries(sidebarCatalog)) {
    if (!id.startsWith(sidebarCategoryPrefix)) continue;
    const englishLabel = id.slice(sidebarCategoryPrefix.length);
    if (
      translation.message === englishLabel &&
      !protectedSidebarLabels.has(englishLabel)
    ) {
      failures.push(`${locale} sidebar category remains in English: ${englishLabel}`);
    }
  }
}

function matches(content, expression) {
  return [...content.matchAll(expression)].map((match) => match[0]);
}

function captures(content, expression) {
  return [...content.matchAll(expression)].map((match) => match[1]);
}

function markdownLinkTargets(content) {
  return captures(content, /(?<!!)\[[^\]]*\]\(([^)]+)\)/g).map((target) =>
    target.replace(/^(\.\.?\/[^?#]+)\.mdx?([?#].*)?$/, "$1$2"),
  );
}

function inlineCodeSpans(content) {
  const withoutFences = withoutPreservedBlocks(content);
  return matches(withoutFences, /(?<!`)`[^`]+`(?!`)/g);
}

function withoutPreservedBlocks(content) {
  return content
    .replace(/```[\s\S]*?```/g, "")
    .replace(/~~~[\s\S]*?~~~/g, "")
    .replace(/<ShellCommand\b[^>]*>[\s\S]*?<\/ShellCommand>/g, "");
}

function listMarkers(content) {
  return captures(withoutPreservedBlocks(content), /^(\s*(?:[-+*]|\d+[.)])\s+)/gm);
}

function tableShapes(content) {
  return withoutPreservedBlocks(content)
    .split("\n")
    .filter((line) => /^\s*\|.*\|\s*$/.test(line))
    .map((line) => (line.match(/(?<!\\)\|/g) || []).length);
}

function headingStructures(content) {
  return withoutPreservedBlocks(content)
    .split("\n")
    .filter((line) => /^#{1,6}\s/.test(line))
    .map((line) => ({
      level: line.match(/^#+/)[0].length,
      inlineCode: matches(line, /(?<!`)`[^`]+`(?!`)/g),
      linkTargets: markdownLinkTargets(line),
      tags: captures(line, /<\/?([A-Za-z][A-Za-z0-9.-]*)\b/g),
    }));
}

const forbiddenEnglishNavigationLabels = new Set([
  "Benchmark Your Model",
  "Build a Custom Data Graph",
  "Build Your First Graph",
  "Configure Model Options",
  "Graph",
  "Model",
  "PCIe Co-processing",
  "Plug a Model Into Your Pipeline",
  "Read Detection Boxes from Model Output",
  "Run / Inference",
  "Run a Graph",
  "Run Your First Model",
  "Tensor and Sample",
  "Troubleshooting",
  "Tune Throughput and Queue Depth",
]);

function englishNavigationLabels(content) {
  return [...content.matchAll(/(?<!!)\[([^\]]+)\]\(([^)]+)\)/g)]
    .filter((match) => /^(?:\/|\.\.?\/)/.test(match[2]))
    .map((match) => match[1].trim())
    .filter((label) => forbiddenEnglishNavigationLabels.has(label));
}

function englishImageAltText(content) {
  return [...content.matchAll(/!\[([^\]]+)\]\(([^)]+)\)/g)]
    .filter((match) => /^(?:@site\/|pathname:\/\/\/|\/|\.\.?\/)/.test(match[2]))
    .map((match) => match[1].trim())
    .filter((label) => {
      const words = label.match(/[A-Za-z]{2,}/g) || [];
      return (
        words.length >= 3 &&
        !/[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/.test(label)
      );
    });
}

function frontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  if (!match) return new Map();

  return new Map(
    match[1]
      .split("\n")
      .map((line) => line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/))
      .filter(Boolean)
      .map((entry) => [entry[1], entry[2]]),
  );
}

function compareExact(relativePath, locale, label, sourceItems, localizedItems) {
  if (JSON.stringify(sourceItems) !== JSON.stringify(localizedItems)) {
    failures.push(`${relativePath}: ${locale} changed ${label}`);
  }
}

function compareMultiset(relativePath, locale, label, sourceItems, localizedItems) {
  compareExact(relativePath, locale, label, [...sourceItems].sort(), [...localizedItems].sort());
}

for (const relativePath of sourcePaths) {
  const sourceBuffer = fs.readFileSync(path.join(repoRoot, relativePath));
  const sourceText = sourceBuffer.toString("utf8");
  const sourceHash = crypto.createHash("sha256").update(sourceBuffer).digest("hex");
  const sourceFrontMatter = frontMatter(sourceText);

  for (const locale of locales) {
    const localizedPath = path.join(
      websiteRoot,
      "i18n",
      locale,
      "docusaurus-plugin-content-docs",
      "current",
      relativePath.replace(/^docs\//, ""),
    );

    if (!fs.existsSync(localizedPath)) continue;
    coverage[locale] += 1;

    if (!sourceHashes[locale] || !sourceHashes[locale][relativePath]) {
      failures.push(`${relativePath}: ${locale} translation has no source-hash record`);
    } else if (sourceHashes[locale][relativePath] !== sourceHash) {
      failures.push(`${relativePath}: ${locale} translation is stale; English source changed`);
    }

    const localizedText = fs.readFileSync(localizedPath, "utf8");
    const localizedFrontMatter = frontMatter(localizedText);

    if (/!\[[^\]]*\]\(pathname:\/\/\//.test(localizedText)) {
      failures.push(
        `${relativePath}: ${locale} uses a pathname:/// image that bypasses DOCS_BASE_URL`,
      );
    }

    if (localizedText.includes("__NEAT_PROTECTED_") || /[\uE000-\uF8FF]/.test(localizedText)) {
      failures.push(`${relativePath}: ${locale} contains an internal localization placeholder`);
    }

    compareExact(
      relativePath,
      locale,
      "front-matter keys",
      [...sourceFrontMatter.keys()].sort(),
      [...localizedFrontMatter.keys()].sort(),
    );

    for (const key of ["id", "slug", "sidebar_position", "pagination_next", "pagination_prev"]) {
      if (sourceFrontMatter.has(key) && sourceFrontMatter.get(key) !== localizedFrontMatter.get(key)) {
        failures.push(`${relativePath}: ${locale} changed protected front-matter field ${key}`);
      }
    }

    compareExact(
      relativePath,
      locale,
      "a fenced code block",
      matches(sourceText, /```[\s\S]*?```/g),
      matches(localizedText, /```[\s\S]*?```/g),
    );
    compareExact(
      relativePath,
      locale,
      "a ShellCommand block",
      matches(sourceText, /<ShellCommand\b[^>]*>[\s\S]*?<\/ShellCommand>/g),
      matches(localizedText, /<ShellCommand\b[^>]*>[\s\S]*?<\/ShellCommand>/g),
    );
    compareExact(
      relativePath,
      locale,
      "an import/export declaration",
      matches(sourceText, /^(?:import|export)\s+.*$/gm),
      matches(localizedText, /^(?:import|export)\s+.*$/gm),
    );
    compareMultiset(
      relativePath,
      locale,
      "an inline code span",
      inlineCodeSpans(sourceText),
      inlineCodeSpans(localizedText),
    );
    compareMultiset(
      relativePath,
      locale,
      "a Markdown link target",
      markdownLinkTargets(sourceText),
      markdownLinkTargets(localizedText),
    );
    compareExact(
      relativePath,
      locale,
      "an HTML link target",
      captures(sourceText, /\bhref=["']([^"']+)["']/g),
      captures(localizedText, /\bhref=["']([^"']+)["']/g),
    );
    compareExact(
      relativePath,
      locale,
      "an admonition type",
      captures(sourceText, /^:::(\w+)/gm),
      captures(localizedText, /^:::(\w+)/gm),
    );
    compareExact(
      relativePath,
      locale,
      "heading levels",
      matches(withoutPreservedBlocks(sourceText), /^#{1,6}(?=\s)/gm),
      matches(withoutPreservedBlocks(localizedText), /^#{1,6}(?=\s)/gm),
    );
    compareExact(
      relativePath,
      locale,
      "protected content between headings",
      headingStructures(sourceText),
      headingStructures(localizedText),
    );
    compareExact(
      relativePath,
      locale,
      "list markers",
      listMarkers(sourceText),
      listMarkers(localizedText),
    );
    compareExact(
      relativePath,
      locale,
      "table column shapes",
      tableShapes(sourceText),
      tableShapes(localizedText),
    );
    compareExact(
      relativePath,
      locale,
      "a heading anchor",
      captures(sourceText, /\{#([^}]+)\}/g),
      captures(localizedText, /\{#([^}]+)\}/g),
    );
    compareExact(
      relativePath,
      locale,
      "an HTML/MDX tag sequence",
      captures(sourceText, /<\/?([A-Za-z][A-Za-z0-9.-]*)\b/g),
      captures(localizedText, /<\/?([A-Za-z][A-Za-z0-9.-]*)\b/g),
    );

    for (const label of englishNavigationLabels(localizedText)) {
      failures.push(`${relativePath}: ${locale} left internal navigation label in English: ${label}`);
    }
    for (const label of englishImageAltText(localizedText)) {
      failures.push(`${relativePath}: ${locale} left image alt text in English: ${label}`);
    }
  }
}

const missing = Object.fromEntries(
  locales.map((locale) => [locale, sourcePaths.length - coverage[locale]]),
);

if (requireComplete) {
  for (const locale of locales) {
    if (missing[locale] > 0) {
      failures.push(`${locale}: ${missing[locale]} authored pages remain untranslated`);
    }
  }
}

console.log(`In-scope authored documentation sources: ${sourcePaths.length}`);
console.log(`English-only reference/generated sources: ${excludedSourcePaths.length}`);
for (const locale of locales) {
  console.log(`${locale}: ${coverage[locale]}/${sourcePaths.length} translated; ${missing[locale]} remaining`);
}

if (failures.length > 0) {
  console.error("\nLocalization validation failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("Existing translations preserve protected documentation structure.");
