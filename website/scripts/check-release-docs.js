"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {load, resolve, tokens, root} = require("../release-data");
const literalPattern = /(?<![\w.])v?\d+\.\d+\.\d+(?:\.\d+)?(?:-[\w.-]+)?|release-\d+\.\d+/;

function walk(directory) {
  return fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const name = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(name) : [name];
  });
}

function check(repoRoot = root, {writeReadmes = false} = {}) {
  const values = load(repoRoot);
  const policy = JSON.parse(fs.readFileSync(path.join(repoRoot, "website/release-data/policy.json"), "utf8"));
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "sima-i18n.config.json"), "utf8"));
  const read = file => fs.readFileSync(path.join(repoRoot, file), "utf8");
  const errors = [];
  for (const key of Object.keys(values)) {
    if (!config.canonicalTerms?.includes(`%%release.${key}%%`)) {
      errors.push(`sima-i18n.config.json: protect %%release.${key}%% in canonicalTerms`);
    }
  }
  const inspect = (file, source) => {
    const manual = (policy.manualPaths || []).some(manualPath =>
      file === manualPath || file.endsWith(`/current/${manualPath.slice(5)}`));
    if (manual && tokens(source).length) {
      errors.push(`${file}: manually maintained documentation must use concrete versions, not release tokens`);
    }
    try { resolve(source, values, file); } catch (error) { errors.push(error.message); }
  };
  const translations = file => Object.keys(config.locales).map(locale =>
    `website/i18n/${locale}/docusaurus-plugin-content-docs/current/${file.slice(5)}`);
  // Scan all docs, including imported content, for broken/unknown tokens. Token
  // parity applies even outside designated rolling paths and catches extra keys.
  for (const absolute of walk(path.join(repoRoot, "docs")).filter(file => /\.mdx?$/.test(file))) {
    const file = path.relative(repoRoot, absolute);
    const source = read(file);
    inspect(file, source);
    if (file.startsWith("docs/release-notes/") && tokens(source).length) {
      errors.push(`${file}: release history must remain pinned`);
    }
    for (const translated of translations(file)) {
      if (!fs.existsSync(path.join(repoRoot, translated))) {
        if (tokens(source).length) errors.push(`${translated}: missing translation for release tokens`);
        continue;
      }
      const target = read(translated);
      inspect(translated, target);
      if (tokens(source).join() !== tokens(target).join()) {
        errors.push(`${translated}: release-token set differs from ${file}; expected ${tokens(source).join(", ")}; got ${tokens(target).join(", ")}`);
      } else {
        for (const key of tokens(source)) {
          const token = `%%release.${key}%%`;
          if (source.split(token).length !== target.split(token).length) {
            errors.push(`${translated}: release-token count for ${key} differs from ${file}`);
          }
        }
      }
    }
  }
  const remaining = new Map(policy.literalExceptions.map(entry => {
    if (!entry.reason?.trim() || !Number.isInteger(entry.count) || entry.count < 1) {
      throw new Error(`Invalid literal exception: ${entry.path}`);
    }
    return [JSON.stringify([entry.path, entry.line]), entry.count];
  }));
  for (const source of policy.rollingPaths) {
    for (const file of [source, ...translations(source)]) {
      read(file).split("\n").forEach((line, index) => {
        if (!literalPattern.test(line)) return;
        const key = JSON.stringify([file, line]);
        if ((remaining.get(key) || 0) > 0) remaining.set(key, remaining.get(key) - 1);
        else errors.push(`${file}:${index + 1}: literal release pin in rolling documentation; use a release token or document an exact historical exception`);
      });
    }
  }
  for (const [key, count] of remaining) {
    if (count) errors.push(`Unused literal exception: ${key}; remove or review it`);
  }
  for (const entry of policy.readmes) {
    let content = read(entry.path);
    const expected = resolve(entry.template, values, entry.path);
    const start = `<!-- release-command:start -->`;
    const end = `<!-- release-command:end -->`;
    const block = `${start}\n\n\`\`\`bash\n${expected}\n\`\`\`\n\n${end}`;
    const pattern = /<!-- release-command:start -->[\s\S]*?<!-- release-command:end -->/g;
    const matches = [...content.matchAll(pattern)];
    if (matches.length !== 1) errors.push(`${entry.path}: expected exactly one release-command block`);
    else if (writeReadmes) {
      content = content.replace(pattern, () => block);
      fs.writeFileSync(path.join(repoRoot, entry.path), content);
    } else if (matches[0][0] !== block) {
      errors.push(`${entry.path}: release command is stale; run npm --prefix website run sync:release-readmes`);
    }
    if (/%{1,2}release\b/i.test(content)) errors.push(`${entry.path}: GitHub README must contain concrete versions, not tokens`);
  }
  if (errors.length) throw new Error(errors.join("\n"));
  return values;
}

if (require.main === module) {
  try {
    check(root, {writeReadmes: process.argv.includes("--write-readmes")});
    console.log("Release data, rolling pins, translation tokens, and README commands are valid.");
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
module.exports = {check, walk};
