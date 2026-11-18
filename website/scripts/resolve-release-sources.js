"use strict";

// Prepare one locale of Markdown sources for the Python search indexer.
const fs = require("node:fs");
const yaml = require("js-yaml");
const {load, resolve} = require("../release-data");

function prepareSource(file, content, values) {
  const text = resolve(content, values, file);
  const header = /^---\n(?:([\s\S]*?)\n)?---(?:\n|$)/.exec(text);
  if (!header) return {body: text};

  const data = yaml.load(header[1] ?? "", {filename: file});
  return {
    body: text.slice(header[0].length),
    title: String(data?.title ?? "").trim(),
    slug: String(data?.slug ?? "").trim(),
  };
}

try {
  const sources = JSON.parse(fs.readFileSync(0, "utf8"));
  const values = load();
  const resolved = Object.fromEntries(Object.entries(sources).map(([file, content]) =>
    [file, prepareSource(file, content, values)]));
  process.stdout.write(JSON.stringify(resolved));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
