const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const websiteRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(websiteRoot, "..");
const manifestPath = path.join(websiteRoot, "i18n", "pilot-translations.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const failures = [];

function protectedBlocks(content) {
  return [
    ...(content.match(/```[\s\S]*?```/g) || []),
    ...(content.match(/<ShellCommand\b[^>]*>[\s\S]*?<\/ShellCommand>/g) || []),
  ];
}

for (const [relativePath, expectedHash] of Object.entries(manifest.sources)) {
  const sourcePath = path.join(repoRoot, relativePath);
  const source = fs.readFileSync(sourcePath);
  const sourceText = source.toString("utf8");
  const actualHash = crypto.createHash("sha256").update(source).digest("hex");

  if (actualHash !== expectedHash) {
    failures.push(`${relativePath}: English source changed; refresh translations`);
  }

  for (const locale of manifest.locales) {
    const localizedPath = path.join(
      websiteRoot,
      "i18n",
      locale,
      "docusaurus-plugin-content-docs",
      "current",
      relativePath.replace(/^docs\//, ""),
    );
    if (!fs.existsSync(localizedPath)) {
      failures.push(`${relativePath}: missing ${locale} translation`);
      continue;
    }

    const localizedText = fs.readFileSync(localizedPath, "utf8");
    const expectedBlocks = protectedBlocks(sourceText);
    const localizedBlocks = protectedBlocks(localizedText);
    if (JSON.stringify(localizedBlocks) !== JSON.stringify(expectedBlocks)) {
      failures.push(`${relativePath}: ${locale} changed a protected code or command block`);
    }
  }
}

if (failures.length > 0) {
  console.error("Pilot translation check failed:\n");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(
  `Pilot translations are complete and current: ${Object.keys(manifest.sources).length} pages x ${manifest.locales.length} locales.`,
);
