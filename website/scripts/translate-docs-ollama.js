const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const {spawnSync} = require("child_process");

const websiteRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(websiteRoot, "..");
const manifestPath = path.join(websiteRoot, "i18n", "translation-sources.json");
const localizationScope = JSON.parse(
  fs.readFileSync(path.join(websiteRoot, "i18n", "scope.json"), "utf8"),
);
const canonicalTerms = [
  "Palette Neat Library",
  "Palette Neat",
  "Neat SDK",
  "Neat Library",
  "Neat Examples",
  "Model Compiler",
  "Model Zoo",
  "Modalix",
  "DevKit Sync",
  "DevKit",
  "PyNeat",
  "Insight",
  "Sentinel",
  "LLiMa",
  "Hugging Face",
  "SiMa.ai",
  "GStreamer",
  "Neat",
];

const localeConfig = {
  ko: {
    name: "Korean",
    code: "ko",
    terms: {
      graph: "그래프",
      "graph fragment": "그래프 조각",
      pipeline: "파이프라인",
      runtime: "런타임",
      "model archive": "모델 아카이브",
      artifact: "아티팩트",
      host: "호스트",
      deployment: "배포",
      compilation: "컴파일",
      validation: "검증",
      troubleshooting: "문제 해결",
      "release notes": "릴리스 노트",
      memory: "메모리",
      thread: "스레드",
      tensor: "텐서",
      architecture: "아키텍처",
      "building block": "구성 요소",
      "source of truth": "단일 진실 공급원",
    },
  },
  ja: {
    name: "Japanese",
    code: "ja",
    terms: {
      graph: "グラフ",
      "graph fragment": "グラフフラグメント",
      pipeline: "パイプライン",
      runtime: "ランタイム",
      "model archive": "モデルアーカイブ",
      artifact: "アーティファクト",
      host: "ホスト",
      deployment: "デプロイ",
      compilation: "コンパイル",
      validation: "検証",
      troubleshooting: "トラブルシューティング",
      "release notes": "リリースノート",
      memory: "メモリ",
      thread: "スレッド",
      tensor: "テンソル",
      architecture: "アーキテクチャ",
      "building block": "構成要素",
      "source of truth": "信頼できる唯一の情報源",
    },
  },
  "zh-Hant": {
    name: "Traditional Chinese (Taiwan)",
    code: "zh-TW",
    terms: {
      graph: "圖",
      "graph fragment": "Graph 片段",
      pipeline: "管線",
      runtime: "執行階段",
      "model archive": "模型封存檔",
      artifact: "成品",
      host: "主機",
      deployment: "部署",
      compilation: "編譯",
      validation: "驗證",
      troubleshooting: "疑難排解",
      "release notes": "版本資訊",
      memory: "記憶體",
      thread: "執行緒",
      tensor: "張量",
      architecture: "架構",
      "building block": "建構模組",
      "source of truth": "權威來源",
    },
    proseReplacements: [
      ["默認值", "預設值"],
      ["插件", "外掛程式"],
      ["配置", "設定"],
      ["構建", "建置"],
      ["加載", "載入"],
      ["支持", "支援"],
      ["默認", "預設"],
      ["數據", "資料"],
      ["鏈接", "連結"],
      ["用戶", "使用者"],
      ["優化", "最佳化"],
      ["信息", "資訊"],
      ["返回", "傳回"],
      ["創建", "建立"],
      ["文件", "檔案"],
      ["線程", "執行緒"],
    ],
  },
  uk: {
    name: "Ukrainian",
    code: "uk-UA",
    terms: {
      graph: "граф",
      "graph fragment": "фрагмент графа",
      pipeline: "конвеєр",
      runtime: "середовище виконання",
      "model archive": "архів моделі",
      artifact: "артефакт",
      host: "хост",
      deployment: "розгортання",
      compilation: "компіляція",
      validation: "перевірка",
      troubleshooting: "усунення несправностей",
      "release notes": "примітки до випуску",
      memory: "пам’ять",
      thread: "потік",
      tensor: "тензор",
      architecture: "архітектура",
      application: "застосунок",
      "building block": "складова",
      folder: "тека",
      "source of truth": "джерело істини",
    },
  },
};

function parseArgs(argv) {
  const args = {
    model: "translategemma:27b",
    sshHost: "macstudio",
    write: false,
    overwrite: false,
    normalizeExisting: false,
    all: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--write") args.write = true;
    else if (argument === "--overwrite") args.overwrite = true;
    else if (argument === "--normalize-existing") args.normalizeExisting = true;
    else if (argument === "--all") args.all = true;
    else if (argument === "--locale") args.locale = argv[++index];
    else if (argument === "--source") args.source = argv[++index];
    else if (argument === "--prefix") args.prefix = argv[++index];
    else if (argument === "--model") args.model = argv[++index];
    else if (argument === "--ssh-host") args.sshHost = argv[++index];
    else throw new Error(`Unknown argument: ${argument}`);
  }

  if (!localeConfig[args.locale]) {
    throw new Error("--locale must be one of: ko, ja, zh-Hant, uk");
  }
  const selectors = [args.all, Boolean(args.source), Boolean(args.prefix)].filter(Boolean).length;
  if (selectors === 0) {
    throw new Error("Provide --source docs/path.md, --prefix docs/section/, or --all");
  }
  if (selectors > 1) {
    throw new Error("Use only one of --source, --prefix, or --all");
  }
  if (args.overwrite && !args.write) {
    throw new Error("--overwrite is valid only with --write");
  }
  if (args.normalizeExisting && !args.write) {
    throw new Error("--normalize-existing is valid only with --write");
  }
  if (args.normalizeExisting && args.overwrite) {
    throw new Error("Do not combine --normalize-existing with --overwrite");
  }
  return args;
}

function trackedSources() {
  const result = spawnSync(
    "git",
    ["ls-files", "-z", "--", "docs/**/*.md", "docs/**/*.mdx"],
    {cwd: repoRoot, encoding: "utf8"},
  );
  if (result.status !== 0) throw new Error(result.stderr || "git ls-files failed");
  return result.stdout
    .split("\0")
    .filter(Boolean)
    .filter(
      (sourcePath) =>
        !localizationScope.excludedPrefixes.some((prefix) => sourcePath.startsWith(prefix)),
    )
    .sort();
}

function localizedPath(locale, sourcePath) {
  return path.join(
    websiteRoot,
    "i18n",
    locale,
    "docusaurus-plugin-content-docs",
    "current",
    sourcePath.replace(/^docs\//, ""),
  );
}

function ollamaChat({sshHost, model, prompt}) {
  const payload = JSON.stringify({
    model,
    stream: false,
    options: {temperature: 0},
    messages: [{role: "user", content: prompt}],
  });
  const result = spawnSync(
    "ssh",
    [
      "-T",
      "-o",
      "BatchMode=yes",
      "-o",
      "ConnectTimeout=10",
      sshHost,
      "curl --silent --show-error --fail http://127.0.0.1:11434/api/chat --header 'Content-Type: application/json' --data-binary @-",
    ],
    {input: payload, encoding: "utf8", maxBuffer: 32 * 1024 * 1024},
  );

  if (result.status !== 0) {
    throw new Error(result.stderr || `Remote Ollama request failed with status ${result.status}`);
  }

  const response = JSON.parse(result.stdout);
  if (!response.message || typeof response.message.content !== "string") {
    throw new Error(`Unexpected Ollama response: ${result.stdout.slice(0, 500)}`);
  }
  return response.message.content.trim();
}

function maskProtectedText(text, translateLinkLabel) {
  if (text.includes("__NEAT_PROTECTED_")) {
    throw new Error("Source text collides with the localization placeholder format");
  }

  const protectedItems = [];
  const hold = (value, hint = "") => {
    const safeHint = hint
      .replace(/[^A-Za-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 48);
    const token = `__NEAT_PROTECTED_${String(protectedItems.length).padStart(4, "0")}${safeHint ? `_${safeHint}` : ""}__`;
    protectedItems.push({token, value});
    return token;
  };

  let masked = text;
  masked = masked.replace(
    /(!?)\[([^\]]*)\]\(([^)]+)\)/g,
    (_, imageMarker, label, target) => {
      const translatedLabel = translateLinkLabel(label);
      return hold(`${imageMarker}[${translatedLabel}](${target})`, label);
    },
  );
  masked = masked.replace(/(?<!`)`([^`]+)`(?!`)/g, (codeSpan, code) => hold(codeSpan, code));
  masked = masked.replace(/<\/?[A-Za-z][^>\n]*>/g, (value) => hold(value));
  masked = masked.replace(/\{[^{}\n]+\}/g, (value) => hold(value));
  masked = masked.replace(/^:::\w+/gm, (value) => hold(value));
  masked = masked.replace(/https?:\/\/[^\s)>]+/g, (value) => hold(value));
  const canonicalPattern = new RegExp(
    canonicalTerms
      .toSorted((left, right) => right.length - left.length)
      .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|"),
    "g",
  );
  masked = masked
    .split(/(__NEAT_PROTECTED_\d+(?:_[A-Za-z0-9_]+)?__)/g)
    .map((part) =>
      part.startsWith("__NEAT_PROTECTED_")
        ? part
        : part.replace(canonicalPattern, (term) => hold(term, term)),
    )
    .join("");

  return {
    masked,
    tokens: protectedItems.map(({token}) => token),
    restore(translated) {
      let restored = translated;
      for (const {token, value} of protectedItems) {
        const count = restored.split(token).length - 1;
        if (count !== 1) {
          throw new Error(
            `Translation changed placeholder ${token} (found ${count} times). Output: ${translated.slice(0, 500)}`,
          );
        }
        restored = restored.replace(token, value);
      }
      if (restored.includes("__NEAT_PROTECTED_")) {
        throw new Error("Translation returned an unknown localization placeholder");
      }
      return restored;
    },
  };
}

function translationPrompt(locale, maskedText, tokens, requiredNames, attempt, fragment) {
  const target = localeConfig[locale];
  const requiredTerms = Object.entries(target.terms)
    .filter(([english]) => maskedText.toLowerCase().includes(english))
    .map(([english, localized]) => `${english} = ${localized}`);
  return [
    `You are a professional English (en) to ${target.name} (${target.code}) translator.`,
    `Your goal is to accurately convey the meaning and nuances of the original English text while adhering to ${target.name} grammar, vocabulary, and cultural sensitivities.`,
    "Preserve every token beginning with __NEAT_PROTECTED_ exactly, including its spelling and position.",
    `Required token checklist: ${tokens.join(", ") || "none"}. Each listed token must appear exactly once in the translation.`,
    attempt > 0 ? "A previous attempt omitted protected content. Translate completely without summarizing or dropping any listed token." : "",
    fragment ? "This is an incomplete sentence fragment next to immutable technical text. Translate only the visible fragment. Do not complete it, add an example, or add Markdown." : "",
    "Preserve Markdown structure, list nesting, headings, tables, whitespace-sensitive directives, and line breaks.",
    requiredNames.length > 0 ? `Keep these product names exactly in English: ${requiredNames.join(", ")}.` : "",
    requiredTerms.length > 0 ? `Use these required technical terms: ${requiredTerms.join("; ")}.` : "",
    `Produce only the ${target.name} translation, without any additional explanations or commentary.`,
    `Please translate the following English text into ${target.name}:`,
    "",
    "",
    maskedText,
  ].join("\n");
}

function translateSegment(args, text, {plain = false, fragment = false} = {}) {
  if (!/[A-Za-z]/.test(text)) return text;
  const leading = text.match(/^\s*/)[0];
  const trailing = text.match(/\s*$/)[0];
  const core = text.slice(leading.length, text.length - trailing.length || undefined);
  if (!core) return text;

  const nonCanonicalText = canonicalTerms.reduce(
    (remaining, term) => remaining.replaceAll(term, ""),
    core,
  );
  if (!/[A-Za-z]/.test(nonCanonicalText)) return text;

  if (/^:::(?:\w+)?/m.test(core)) {
    const translatedDirective = core
      .split(/(^:::\w+[^\n]*$|^:::$)/gm)
      .map((part) => {
        if (part === ":::") return part;
        const opening = part.match(/^(:::\w+)(.*)$/);
        if (opening) return `${opening[1]}${translateSegment(args, opening[2])}`;
        return translateSegment(args, part);
      })
      .join("");
    return `${leading}${translatedDirective}${trailing}`;
  }

  const inlineCode = [];
  const htmlSafeCore = core.replace(/(?<!`)`[^`]+`(?!`)/g, (codeSpan) => {
    const marker = `\uE000${inlineCode.length}\uE001`;
    inlineCode.push(codeSpan);
    return marker;
  });
  if (/<\/?[A-Za-z][^>\n]*>/.test(htmlSafeCore)) {
    const translatedNodes = htmlSafeCore
      .split(/(<\/?[A-Za-z][^>\n]*>)/g)
      .map((part) => {
        if (part.startsWith("<") && part.endsWith(">")) return part;
        const restoredCode = part.replace(/\uE000(\d+)\uE001/g, (_, index) => inlineCode[Number(index)]);
        return translateSegment(args, restoredCode);
      })
      .join("");
    return `${leading}${translatedNodes}${trailing}`;
  }

  const {masked, restore, tokens} = maskProtectedText(core, (label) => translateSegment(args, label));
  const unprotectedText = masked.replace(/__NEAT_PROTECTED_\d+(?:_[A-Za-z0-9_]+)?__/g, "");
  if (!/[A-Za-z]/.test(unprotectedText)) return text;
  const requiredNames = canonicalTerms.filter((term) => core.includes(term));

  let lastError;
  const maximumAttempts = 3;
  for (let attempt = 0; attempt < maximumAttempts; attempt += 1) {
    const translated = ollamaChat({
      sshHost: args.sshHost,
      model: args.model,
      prompt: translationPrompt(args.locale, masked, tokens, requiredNames, attempt, fragment),
    });
    try {
      let restored = restore(translated);
      if (plain && !core.includes("`")) {
        restored = restored.replace(/`([^`]+)`/g, "$1");
      }
      const sourceInlineCode = [...core.matchAll(/(?<!`)`[^`]+`(?!`)/g)]
        .map((match) => match[0])
        .sort();
      if (sourceInlineCode.length === 0) {
        restored = restored.replace(/`([^`]+)`/g, "$1");
      }
      const translatedInlineCode = [...restored.matchAll(/(?<!`)`[^`]+`(?!`)/g)]
        .map((match) => match[0])
        .sort();
      if (JSON.stringify(translatedInlineCode) !== JSON.stringify(sourceInlineCode)) {
        throw new Error(
          `Translation added or changed inline code. Output: ${translated.slice(0, 500)}`,
        );
      }
      for (const term of requiredNames) {
        const sourceCount = core.split(term).length - 1;
        const translatedCount = restored.split(term).length - 1;
        if (translatedCount < sourceCount) {
          throw new Error(
            `Translation omitted canonical term ${term} (expected at least ${sourceCount}, found ${translatedCount}). Output: ${translated.slice(0, 500)}`,
          );
        }
      }
      return `${leading}${restored}${trailing}`;
    } catch (error) {
      lastError = error;
    }
  }
  if (tokens.length > 0) {
    const tokenPattern = /(__NEAT_PROTECTED_\d+(?:_[A-Za-z0-9_]+)?__)/g;
    const translatedAroundProtectedItems = masked
      .split(tokenPattern)
      .map((part) => (tokens.includes(part) ? part : translateSegment(args, part, {plain, fragment: true})))
      .join("");
    let restored = restore(translatedAroundProtectedItems);
    if (plain && !core.includes("`")) restored = restored.replace(/`([^`]+)`/g, "$1");
    return `${leading}${restored}${trailing}`;
  }
  throw lastError;
}

function translateFrontMatter(args, source) {
  const match = source.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) return {frontMatter: "", body: source};

  const translated = match[1]
    .split("\n")
    .map((line) => {
      const field = line.match(/^(title|description|sidebar_label):\s*(.+)$/);
      if (!field) return line;
      return `${field[1]}: ${translateSegment(args, field[2], {plain: true})}`;
    })
    .join("\n");
  return {
    frontMatter: `---\n${translated}\n---\n`,
    body: source.slice(match[0].length),
  };
}

function shouldPreserveBlock(block) {
  const trimmed = block.trimStart();
  return (
    trimmed.startsWith("```") ||
    trimmed.startsWith("~~~") ||
    trimmed.startsWith("<ShellCommand") ||
    /^(?:import|export)\s/m.test(trimmed)
  );
}

function rewriteLocalizedImages(content, sourcePath) {
  const assetRoots = ["docs/images/", "docs/develop-apps/images/"];
  return content.replace(/(!\[[^\]]*\]\()([^)\s]+)(\))/g, (match, opening, target, closing) => {
    if (/^(?:[a-z]+:|\/)/i.test(target)) return match;
    const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), target));
    const assetRoot = assetRoots.find((root) => resolved.startsWith(root));
    if (!assetRoot) return match;
    // Resolve through webpack so Docusaurus applies DOCS_BASE_URL. The old
    // pathname:/// form deliberately bypassed baseUrl and broke images when
    // Software docs were mounted below /software/ in the Developer Center.
    return `${opening}@site/../${resolved}${closing}`;
  });
}

function normalizeLocalizedProse(locale, content) {
  const replacements = localeConfig[locale].proseReplacements || [];
  const protectedItems = [];
  const hold = (value) => {
    const token = `__NEAT_NORMALIZE_${String(protectedItems.length).padStart(4, "0")}__`;
    protectedItems.push({token, value});
    return token;
  };
  let normalized = content.replace(
    /```[\s\S]*?```|~~~[\s\S]*?~~~|<ShellCommand\b[^>]*>[\s\S]*?<\/ShellCommand>|(?<!`)`[^`]+`(?!`)/g,
    (value) => hold(value),
  );
  for (const [source, target] of replacements) {
    normalized = normalized.replaceAll(source, target);
  }
  for (const {token, value} of protectedItems) {
    normalized = normalized.replace(token, value);
  }
  return normalized.replace(
    /^(title|description|sidebar_label):\s*(.*)$/gm,
    (_, key, value) => {
      const trimmed = value.trim();
      if (/^"(?:[^"\\]|\\.)*"$/.test(trimmed)) return `${key}: ${trimmed}`;
      return `${key}: ${JSON.stringify(trimmed)}`;
    },
  );
}

function chunkProse(part, maximumCharacters = 3000) {
  const pieces = part.split(/(\n{2,})/);
  const chunks = [];
  let current = "";

  for (const piece of pieces) {
    if (current && current.length + piece.length > maximumCharacters && !/^\n+$/.test(piece)) {
      chunks.push(current);
      current = "";
    }
    current += piece;
  }
  if (current) chunks.push(current);
  return chunks;
}

function translateTableRow(args, line) {
  if (/^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$/.test(line)) return line;

  const separators = [];
  let masked = "";
  let inCode = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === "`" && line[index - 1] !== "\\") inCode = !inCode;
    if (character !== "|" || inCode || line[index - 1] === "\\") {
      masked += character;
      continue;
    }
    const token = `<NeatTableSeparator data-index="${separators.length}" />`;
    separators.push(token);
    masked += token;
  }

  let translated = translateSegment(args, masked);
  for (const token of separators) {
    if (translated.split(token).length - 1 !== 1) {
      throw new Error(`Translation changed a protected table separator: ${translated.slice(0, 500)}`);
    }
    translated = translated.replace(token, "|");
  }
  return translated;
}

function structuralLineSignature(content) {
  return [...content.matchAll(/^(\s*(?:#{1,6}\s+|(?:[-+*]|\d+[.)])\s+|>\s?|\|.*\|\s*$))/gm)]
    .map((match) => match[1]);
}

function translateOrdinaryChunk(args, chunk) {
  const translated = translateSegment(args, chunk);
  if (
    JSON.stringify(structuralLineSignature(translated)) ===
    JSON.stringify(structuralLineSignature(chunk))
  ) {
    return translated;
  }

  const constrained = chunk
    .split(/(\n)/)
    .map((part) => (part === "\n" || part.trim() === "" ? part : translateSegment(args, part, {fragment: true})))
    .join("");
  if (
    JSON.stringify(structuralLineSignature(constrained)) !==
    JSON.stringify(structuralLineSignature(chunk))
  ) {
    throw new Error(`Translation changed Markdown block structure: ${constrained.slice(0, 500)}`);
  }
  return constrained;
}

function translateStructuredPart(args, part) {
  const output = [];
  let ordinary = "";
  const flushOrdinary = () => {
    if (!ordinary) return;
    output.push(
      chunkProse(ordinary)
        .map((chunk) => (shouldPreserveBlock(chunk) ? chunk : translateOrdinaryChunk(args, chunk)))
        .join(""),
    );
    ordinary = "";
  };

  for (const match of part.matchAll(/([^\n]*)(\n|$)/g)) {
    const line = match[1];
    const newline = match[2];
    if (!line && !newline) continue;
    const prefix = line.match(/^(\s*(?:#{1,6}\s+|(?:[-+*]|\d+[.)])\s+|>\s?))(.*)$/);
    const table = /^\s*\|.*\|\s*$/.test(line);
    if (!prefix && !table) {
      ordinary += `${line}${newline}`;
      continue;
    }

    flushOrdinary();
    if (table) output.push(`${translateTableRow(args, line)}${newline}`);
    else output.push(`${prefix[1]}${translateSegment(args, prefix[2])}${newline}`);
  }
  flushOrdinary();
  return output.join("");
}

function translateDocument(args, source, sourcePath) {
  const {frontMatter, body} = translateFrontMatter(args, source);
  const fenced = body.split(/(```[\s\S]*?```|~~~[\s\S]*?~~~|<ShellCommand\b[^>]*>[\s\S]*?<\/ShellCommand>)/g);
  const translated = fenced
    .map((part) => {
      if (!part || shouldPreserveBlock(part)) return part;
      return translateStructuredPart(args, part);
    })
    .join("");
  return normalizeLocalizedProse(
    args.locale,
    rewriteLocalizedImages(`${frontMatter}${translated}`, sourcePath),
  );
}

function writeManifest(manifest) {
  for (const locale of Object.keys(manifest)) {
    manifest[locale] = Object.fromEntries(Object.entries(manifest[locale]).sort(([a], [b]) => a.localeCompare(b)));
  }
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const allSources = trackedSources();
  const selected = args.all
    ? allSources
    : args.prefix
      ? allSources.filter((sourcePath) => sourcePath.startsWith(args.prefix))
      : [args.source];
  if (selected.length === 0) throw new Error(`No authored documentation matched ${args.prefix}`);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  let translatedCount = 0;

  for (const sourcePath of selected) {
    if (!allSources.includes(sourcePath)) throw new Error(`${sourcePath} is not a tracked authored documentation page`);
    const targetPath = localizedPath(args.locale, sourcePath);
    if (args.normalizeExisting) {
      if (!fs.existsSync(targetPath)) {
        console.log(`skip ${sourcePath}: ${args.locale} translation does not exist`);
        continue;
      }
      const current = fs.readFileSync(targetPath, "utf8");
      const normalized = normalizeLocalizedProse(args.locale, current);
      if (normalized === current) {
        console.log(`skip ${sourcePath}: ${args.locale} terminology already normalized`);
        continue;
      }
      const temporaryPath = `${targetPath}.tmp`;
      fs.writeFileSync(temporaryPath, normalized);
      fs.renameSync(temporaryPath, targetPath);
      translatedCount += 1;
      console.log(`normalized ${sourcePath} -> ${args.locale}`);
      continue;
    }
    if (fs.existsSync(targetPath) && !args.overwrite) {
      console.log(`skip ${sourcePath}: ${args.locale} translation already exists`);
      continue;
    }

    const sourceBuffer = fs.readFileSync(path.join(repoRoot, sourcePath));
    const localized = translateDocument(args, sourceBuffer.toString("utf8"), sourcePath);
    if (!localized.endsWith("\n")) throw new Error(`${sourcePath}: generated translation has no final newline`);

    if (!args.write) {
      process.stdout.write(localized);
      if (selected.length > 1) process.stdout.write("\n");
      continue;
    }

    fs.mkdirSync(path.dirname(targetPath), {recursive: true});
    const temporaryPath = `${targetPath}.tmp`;
    fs.writeFileSync(temporaryPath, localized);
    fs.renameSync(temporaryPath, targetPath);
    const hash = crypto.createHash("sha256").update(sourceBuffer).digest("hex");
    manifest[args.locale][sourcePath] = hash;
    writeManifest(manifest);
    translatedCount += 1;
    console.log(`translated ${sourcePath} -> ${args.locale}`);
  }

  if (args.write) console.log(`Wrote ${translatedCount} translation(s).`);
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
