const fs = require("node:fs");
const path = require("node:path");
const {createSlugger, parseMarkdownHeadingId} = require("@docusaurus/utils");

// Imported translations preserve English anchors; other portal pages are untouched.
module.exports = function kernelHeadings() {
  const catalogs = new Map();
  return (tree, file) => {
    const normalized = file.path.split(path.sep).join("/");
    const marker = "/kernel-library/";
    const offset = normalized.lastIndexOf(marker);
    if (offset < 0) return;
    const root = normalized.slice(0, offset + marker.length);
    const catalog = path.join(root, "assets/canonical-headings.json");
    if (!fs.existsSync(catalog)) return; // English and generated reference pages.
    if (!catalogs.has(catalog)) catalogs.set(catalog, JSON.parse(fs.readFileSync(catalog, "utf8")));
    const expected = catalogs.get(catalog)[normalized.slice(offset + 1)];
    if (!expected) return;
    const nodes = [];
    function collect(node) {
      if (node.type === "heading") nodes.push(node);
      if (node.children) node.children.forEach(collect);
    }
    collect(tree);
    if (nodes.length !== expected.length) throw new Error(`Kernel heading count changed: ${file.path}`);
    const slugs = createSlugger();
    nodes.forEach((node, index) => {
      const original = expected[index];
      if (node.depth !== original.depth) throw new Error(`Kernel heading depth changed: ${file.path}`);
      if (index === 0 && node.depth === 1) return;
      const parsed = parseMarkdownHeadingId(original.text, "classic");
      node.data ||= {};
      node.data.hProperties ||= {};
      node.data.hProperties.id = parsed.id || slugs.slug(parsed.text);
    });
  };
};
