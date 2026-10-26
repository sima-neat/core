"use strict";

const fs = require("node:fs");
const path = require("node:path");
const root = path.resolve(__dirname, "../..");
const keys = ["core", "platform", "sdk", "sdk_channel", "sdk_series", "apps", "model_compiler", "insight", "sima_cli"];
const tokenPattern = /%%release\.([a-z][a-z0-9_]*)%%/g;

function validate(data, manifest) {
  if (!data || data.schemaVersion !== 1 || !data.releases || typeof data.releases !== "object") {
    throw new Error("release-data/current.json: expected schemaVersion 1 and releases object");
  }
  if (Object.keys(data.releases).sort().join() !== [...keys].sort().join()) {
    throw new Error(`release-data/current.json: expected exactly these release keys: ${keys.join(", ")}`);
  }
  const values = {};
  for (const key of keys) {
    const entry = data.releases[key];
    const pattern = key === "sdk_channel" ? /^release-\d+\.\d+$/
      : key === "sdk_series" ? /^\d+\.\d+$/ : /^\d+\.\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*)?$/;
    if (!entry || typeof entry.value !== "string" || !pattern.test(entry.value) || entry.value.trim() !== entry.value ||
        typeof entry.owner !== "string" || !entry.owner.trim() ||
        typeof entry.source !== "string" || !entry.source.trim()) {
      throw new Error(`release-data/current.json: malformed ${key}; provide a valid value, owner, and source`);
    }
    values[key] = entry.value;
  }
  for (const [key, field] of [["core", "package-version"], ["platform", "platform-version"]]) {
    if (values[key] !== manifest[field]) {
      throw new Error(`release-data/current.json: ${key}=${values[key]} differs from deps/manifest.json ${field}=${manifest[field]}`);
    }
  }
  if (values.sdk_channel !== `release-${values.sdk_series}` || !values.sdk.startsWith(`${values.sdk_series}.`)) {
    throw new Error("release-data/current.json: sdk, sdk_series, and sdk_channel must describe the same release series");
  }
  return values;
}

function load(repoRoot = root) {
  return validate(
    JSON.parse(fs.readFileSync(path.join(repoRoot, "website/release-data/current.json"), "utf8")),
    JSON.parse(fs.readFileSync(path.join(repoRoot, "deps/manifest.json"), "utf8")),
  );
}

function resolve(content, values, file = "document") {
  const result = content.replace(/%%release\.([a-z][a-z0-9_]*)%%/g, (token, key, offset) => {
    if (!Object.hasOwn(values, key)) {
      throw new Error(`${file}:${content.slice(0, offset).split("\n").length}: unknown release key ${key}`);
    }
    return values[key];
  });
  const leftover = /%{1,2}release\b/i.exec(result);
  if (leftover) {
    throw new Error(`${file}:${result.slice(0, leftover.index).split("\n").length}: malformed or unresolved release token near ${result.slice(leftover.index, leftover.index + 60)}`);
  }
  return result;
}

function tokens(content) {
  return [...new Set([...content.matchAll(tokenPattern)].map(match => match[1]))].sort();
}

// Embed the validated snapshot in the function's source. Docusaurus hashes the
// preprocessor in its loader options: changing only release data invalidates
// compiled Markdown caches, including when building without build.sh's cleanup.
function preprocessor(values) {
  return new Function("input", `return (${resolve.toString()})(input.fileContent, ${JSON.stringify(values)}, input.filePath);`);
}

module.exports = {load, validate, resolve, tokens, preprocessor, root};
