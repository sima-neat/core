"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const sourcePath = path.resolve(
  __dirname,
  "../src/clientModules/analytics-consent.js",
);
const source = fs
  .readFileSync(sourcePath, "utf8")
  .replace(/^import .*;\n/, "")
  .replace(/^export /gm, "")
  .concat(
    "\nglobalThis.__analyticsTest = " +
      "{normalizedDocsPath, docSectionFromPath, tutorialIdFromPath};\n",
  );
const context = {
  ExecutionEnvironment: {canUseDOM: false},
  URL,
  URLSearchParams,
  window: {
    __NEAT_DOCS_ANALYTICS__: {
      baseUrl: "/software/",
      locales: ["en", "ko", "ja", "zh-Hant", "uk"],
    },
    location: {href: "https://developer.sima.ai/software/"},
  },
};
vm.runInNewContext(source, context, {filename: sourcePath});

const {normalizedDocsPath, docSectionFromPath, tutorialIdFromPath} =
  context.__analyticsTest;

assert.equal(
  normalizedDocsPath("/software/ja/tutorials/run-your-first-model/"),
  "/tutorials/run-your-first-model",
);
assert.equal(
  normalizedDocsPath("/software/zh-Hant/reference/cppapi/Graph"),
  "/reference/cppapi/Graph",
);
assert.equal(
  normalizedDocsPath("/software/getting-started/neat-library"),
  "/getting-started/neat-library",
);
assert.equal(
  docSectionFromPath("/software/uk/reference/pythonapi/Tensor"),
  "api-reference-python",
);
assert.equal(
  docSectionFromPath("/software/ko/getting-started/neat-library"),
  "getting-started",
);
assert.equal(
  tutorialIdFromPath("/software/ja/tutorials/run-your-first-model/"),
  "run-your-first-model",
);

console.log("Localized analytics routes normalize to canonical documentation paths.");
