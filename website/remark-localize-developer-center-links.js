"use strict";

const DEVELOPER_CENTER_HOSTS = new Set([
  "developer.sima.ai",
  "developer-stg.sima.ai",
]);
const DOCS_LOCALES = new Set(["ko", "ja", "zh-Hant", "uk"]);
const LOCALE_BEFORE_PILLARS = new Set(["examples", "hardware"]);

function localeForFile(filePath) {
  const normalized = String(filePath || "").replaceAll("\\", "/");
  const match = normalized.match(
    /(?:^|\/)i18n\/([^/]+)\/docusaurus-plugin-content-docs(?:\/|$)/,
  );
  return match && DOCS_LOCALES.has(match[1]) ? match[1] : "";
}

function localizeDeveloperCenterUrl(rawUrl, locale, siteUrl) {
  if (!DOCS_LOCALES.has(locale)) return rawUrl;

  let url;
  try {
    url = new URL(rawUrl);
  } catch {
    return rawUrl;
  }
  if (!DEVELOPER_CENTER_HOSTS.has(url.hostname)) return rawUrl;

  const segments = url.pathname.split("/").filter(Boolean);
  if (segments[0] === "software") {
    if (DOCS_LOCALES.has(segments[1])) {
      segments[1] = locale;
    } else {
      segments.splice(1, 0, locale);
    }
  } else if (LOCALE_BEFORE_PILLARS.has(segments[0])) {
    segments.unshift(locale);
  } else if (
    DOCS_LOCALES.has(segments[0]) &&
    LOCALE_BEFORE_PILLARS.has(segments[1])
  ) {
    segments[0] = locale;
  } else {
    return rawUrl;
  }

  const targetOrigin = new URL(siteUrl || url.origin).origin;
  const trailingSlash = url.pathname.endsWith("/") ? "/" : "";
  return `${targetOrigin}/${segments.join("/")}${trailingSlash}${url.search}${url.hash}`;
}

function visit(node, visitor) {
  if (!node || typeof node !== "object") return;
  visitor(node);
  for (const child of node.children || []) visit(child, visitor);
}

function remarkLocalizeDeveloperCenterLinks(options = {}) {
  return (tree, file) => {
    const locale = localeForFile(file?.path || file?.history?.[0]);
    if (!locale) return;

    visit(tree, (node) => {
      if ((node.type === "link" || node.type === "definition") && node.url) {
        node.url = localizeDeveloperCenterUrl(node.url, locale, options.siteUrl);
      }

      if (node.type === "mdxJsxFlowElement" || node.type === "mdxJsxTextElement") {
        for (const attribute of node.attributes || []) {
          if (attribute.name === "href" && typeof attribute.value === "string") {
            attribute.value = localizeDeveloperCenterUrl(
              attribute.value,
              locale,
              options.siteUrl,
            );
          }
        }
      }
    });
  };
}

module.exports = remarkLocalizeDeveloperCenterLinks;
module.exports.localeForFile = localeForFile;
module.exports.localizeDeveloperCenterUrl = localizeDeveloperCenterUrl;
