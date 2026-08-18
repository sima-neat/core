const fs = require("node:fs");
const path = require("node:path");

const outputDir = path.resolve(
  __dirname,
  "..",
  process.env.INSIGHT_API_OUTPUT || "../docs/tools/insight/api",
);
const expectedSuffix = path.join("tools", "insight", "api");

if (!outputDir.endsWith(expectedSuffix)) {
  throw new Error(
    `Refusing to clean unexpected Insight API output directory: ${outputDir}`,
  );
}

fs.rmSync(outputDir, {recursive: true, force: true});
