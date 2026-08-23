"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const remarkLocalizeDeveloperCenterLinks = require(
  "../remark-localize-developer-center-links.js",
);
const {
  localeForFile,
  localizeDeveloperCenterUrl,
} = remarkLocalizeDeveloperCenterLinks;

const websiteRoot = path.resolve(__dirname, "..");
const translatedDocsRoot = path.join(websiteRoot, "i18n");
const locales = ["ko", "ja", "zh-Hant", "uk"];
const developerCenterUrl = /https:\/\/developer(?:-stg)?\.sima\.ai\/[^)\s>"\]]+/g;

assert.equal(
  localeForFile(
    "/repo/website/i18n/zh-Hant/docusaurus-plugin-content-docs/current/page.md",
  ),
  "zh-Hant",
);
assert.equal(localeForFile("/repo/docs/page.md"), "");

for (const [source, expected] of [
  [
    "https://developer.sima.ai/hardware/getting-started/setup-serial",
    "https://developer-stg.sima.ai/zh-Hant/hardware/getting-started/setup-serial",
  ],
  [
    "https://developer.sima.ai/examples",
    "https://developer-stg.sima.ai/zh-Hant/examples",
  ],
  [
    "https://developer.sima.ai/software/getting-started/dev-environment/?tab=one#setup",
    "https://developer-stg.sima.ai/software/zh-Hant/getting-started/dev-environment/?tab=one#setup",
  ],
  [
    "https://developer.sima.ai/ja/hardware/getting-started/setup-serial",
    "https://developer-stg.sima.ai/zh-Hant/hardware/getting-started/setup-serial",
  ],
]) {
  assert.equal(
    localizeDeveloperCenterUrl(source, "zh-Hant", "https://developer-stg.sima.ai"),
    expected,
  );
}

assert.equal(
  localizeDeveloperCenterUrl(
    "https://example.com/hardware/getting-started",
    "zh-Hant",
    "https://developer-stg.sima.ai",
  ),
  "https://example.com/hardware/getting-started",
);

const tree = {
  type: "root",
  children: [
    {
      type: "link",
      url: "https://developer.sima.ai/hardware/getting-started/setup-serial",
      children: [],
    },
  ],
};
remarkLocalizeDeveloperCenterLinks({
  siteUrl: "https://developer-stg.sima.ai",
})(tree, {
  path: "/repo/website/i18n/ja/docusaurus-plugin-content-docs/current/page.md",
});
assert.equal(
  tree.children[0].url,
  "https://developer-stg.sima.ai/ja/hardware/getting-started/setup-serial",
);

function markdownFiles(root) {
  const files = [];
  for (const entry of fs.readdirSync(root, {withFileTypes: true})) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) files.push(...markdownFiles(target));
    if (entry.isFile() && /\.mdx?$/.test(entry.name)) files.push(target);
  }
  return files;
}

let localizedLinks = 0;
for (const locale of locales) {
  const localeRoot = path.join(
    translatedDocsRoot,
    locale,
    "docusaurus-plugin-content-docs",
    "current",
  );
  for (const file of markdownFiles(localeRoot)) {
    const source = fs.readFileSync(file, "utf8");
    for (const match of source.matchAll(developerCenterUrl)) {
      const rawUrl = match[0].replace(/[.,]$/, "");
      const localized = localizeDeveloperCenterUrl(
        rawUrl,
        locale,
        "https://developer-stg.sima.ai",
      );
      assert.notEqual(
        localized,
        rawUrl,
        `${file} contains a Developer Center link without a localization rule: ${rawUrl}`,
      );
      localizedLinks += 1;
      const localizedPath = new URL(localized).pathname;
      assert.ok(
        localizedPath.startsWith(`/${locale}/`) ||
          localizedPath.startsWith(`/software/${locale}/`),
        `${file} produced a non-localized Developer Center URL: ${localized}`,
      );
    }
  }
}

assert.ok(localizedLinks > 0, "Expected translated Developer Center links to exercise the rule");
console.log(`Localized Developer Center link checks passed (${localizedLinks} links).`);
