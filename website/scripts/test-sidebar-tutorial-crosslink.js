const assert = require("node:assert/strict");
const test = require("node:test");
const generate = require("../sidebarItemsGenerator");

const tutorialId = "develop-apps/tutorials/pcie/tutorial_029_run_genai_over_pcie";
const categoryId = "develop-apps/tutorials/genai/index";

test("GenAI sidebar refers to the one canonical PCIe tutorial with its localized title", async () => {
  const result = await generate({
    docs: [{id: tutorialId, title: "PCIe 経由で GenAI モデルを実行する"}],
    defaultSidebarItemsGenerator: async () => [{
      type: "category",
      label: "Tutorials",
      items: [{type: "category", label: "GenAI", link: {type: "doc", id: categoryId}, items: []}],
    }],
  });
  const links = result[0].items[0].items;
  assert.deepEqual(links, [{type: "ref", id: tutorialId, label: "PCIe 経由で GenAI モデルを実行する"}]);
});

test("missing tutorial or an existing reference does not add a duplicate", async () => {
  for (const docs of [[], [{id: tutorialId, title: "PCIe GenAI"}]]) {
    const result = await generate({
      docs,
      defaultSidebarItemsGenerator: async () => [{
        type: "category", label: "GenAI", link: {type: "doc", id: categoryId},
        items: [{type: "ref", id: tutorialId}],
      }],
    });
    assert.equal(result[0].items.length, 1);
  }
});
