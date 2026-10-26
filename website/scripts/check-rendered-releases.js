"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {load, root} = require("../release-data");
const {walk} = require("./check-release-docs");

function checkRendered(buildDir = path.join(root, "website/build"), locales = ["en", "ko", "ja", "zh-Hant", "uk"]) {
  const values = load();
  const htmlFiles = walk(buildDir).filter(file => file.endsWith(".html"));
  if (!htmlFiles.length) throw new Error(`${buildDir}: no rendered HTML found`);
  for (const file of htmlFiles) {
    const html = fs.readFileSync(file, "utf8");
    if (/(?:%|%25|&#37;){1,2}release(?:\.|%2e)/i.test(html)) {
      throw new Error(`${file}: unresolved release token in rendered HTML`);
    }
  }
  const examples = [
    ["getting-started/neat-library/pcie-host", `sima-cli neat install core/pciehost/ubuntu24/amd64@v${values.core}`],
    ["getting-started/dev-environment/install-the-environment", `sima-cli neat install sdk@${values.sdk_channel}`],
    ["getting-started/dev-environment/offline-installation", `sima-cli neat install sdk@v${values.sdk} -t offline-amd64`],
    ["getting-started/dev-environment/install-model-compiler", `sima-cli neat install model-compiler/amd64@v${values.model_compiler}`],
  ];
  for (const locale of locales) {
    for (const [route, expected] of examples) {
      const base = path.join(buildDir, locale === "en" ? "" : locale, route);
      const file = [base + ".html", path.join(base, "index.html")].find(fs.existsSync);
      if (!file) throw new Error(`${base}: missing rendered release example`);
      const html = fs.readFileSync(file, "utf8");
      // Syntax highlighting wraps fragments in spans; compare rendered text too.
      if (!html.includes(expected) && !html.replace(/<[^>]*>/g, "").includes(expected)) {
        throw new Error(`${file}: expected concrete release example ${expected}`);
      }
    }
  }
}
if (require.main === module) {
  try {
    checkRendered(process.argv[2], process.argv[3]?.split(","));
    console.log("Rendered release commands are concrete in every requested locale.");
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
module.exports = {checkRendered};
