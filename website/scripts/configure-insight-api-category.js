const fs = require("node:fs");
const path = require("node:path");

const outputDir = path.resolve(
  __dirname,
  "..",
  process.env.INSIGHT_API_OUTPUT || "../docs/tools/insight/api",
);

const category = {
  label: "API",
  link: {
    type: "doc",
    id: "tools/insight/api/neat-insight-api",
  },
};

fs.writeFileSync(
  path.join(outputDir, "_category_.json"),
  `${JSON.stringify(category, null, 2)}\n`,
);

const generatedSidebarPath = path.join(outputDir, "sidebar.ts");
const generatedSidebar = fs
  .readFileSync(generatedSidebarPath, "utf8")
  .replace(/^import type .*;\n+/m, "")
  .replace("const sidebar: SidebarsConfig =", "const sidebar =")
  .replace(
    "export default sidebar.apisidebar;",
    "module.exports = sidebar.apisidebar;",
  );

fs.writeFileSync(path.join(outputDir, "sidebar.js"), generatedSidebar);
