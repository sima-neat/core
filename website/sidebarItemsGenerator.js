const path = require("node:path");

const insightApiOutput = path.resolve(
  __dirname,
  process.env.INSIGHT_API_OUTPUT || "../docs/tools/insight/api",
);

function isApiIntroduction(item) {
  return (
    item.type === "doc" &&
    item.id === "tools/insight/api/neat-insight-api"
  );
}

function removeRepeatedIntroductions(items) {
  return items
    .filter((item) => !isApiIntroduction(item))
    .map((item) =>
      item.type === "category"
        ? {...item, items: removeRepeatedIntroductions(item.items)}
        : item,
    );
}

function replaceApiItems(items, apiItems) {
  return items.map((item) => {
    if (item.type !== "category") return item;

    if (
      item.label === "API" &&
      item.link?.type === "doc" &&
      item.link.id === "tools/insight/api/neat-insight-api"
    ) {
      return {...item, items: apiItems};
    }

    return {...item, items: replaceApiItems(item.items, apiItems)};
  });
}

module.exports = async function sidebarItemsGenerator(args) {
  const generatedItems = await args.defaultSidebarItemsGenerator(args);

  try {
    const generatedApiSidebar = require(path.join(insightApiOutput, "sidebar.js"));
    const apiItems = removeRepeatedIntroductions(generatedApiSidebar);
    return replaceApiItems(generatedItems, apiItems);
  } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND") throw error;
    return generatedItems;
  }
};
