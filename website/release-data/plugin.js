"use strict";

const {checkRendered} = require("../scripts/check-rendered-releases");

module.exports = function releaseOutputChecks() {
  return {
    name: "release-output-checks",
    postBuild({outDir}) {
      // Docusaurus invokes this once per locale and supplies that locale's
      // actual output root. This also supports --locale and --out-dir builds.
      checkRendered(outDir, ["en"]);
      console.log(`Verified concrete release commands in ${outDir}`);
    },
  };
};
