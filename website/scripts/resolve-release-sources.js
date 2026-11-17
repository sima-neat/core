"use strict";

// JSON bridge for non-Docusaurus consumers, including the Python search indexer.
// Resolve the whole batch against the same validated snapshot used by the site.
const fs = require("node:fs");
const {load, resolve} = require("../release-data");

try {
  const sources = JSON.parse(fs.readFileSync(0, "utf8"));
  const values = load();
  const resolved = Object.fromEntries(Object.entries(sources).map(([file, content]) =>
    [file, resolve(content, values, file)]));
  process.stdout.write(JSON.stringify(resolved));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
