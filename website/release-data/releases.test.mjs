import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {evaluate} from '@mdx-js/mdx';
import * as runtime from 'react/jsx-runtime';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
const require = createRequire(import.meta.url);
const {load, validate, resolve, preprocessor, root} = require('./index.js');
const {check} = require('../scripts/check-release-docs.js');
const data = JSON.parse(fs.readFileSync(path.join(root, 'website/release-data/current.json')));
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'deps/manifest.json')));
const values = load();

test('validates schema, all required fields, safe scalar values, ownership, and authoritative versions', () => {
  assert.equal(validate(data, manifest).core, manifest['package-version']);
  for (const mutate of [
    d => { d.schemaVersion = 2; },
    d => { delete d.releases.insight; },
    d => { d.releases.core.value = '9.9.9'; },
    d => { d.releases.platform.value = '9.9.9'; },
    d => { d.releases.sdk.value = '<script>'; },
    d => { d.releases.insight.owner = ''; },
    d => { d.releases.sdk_channel.value = 'release-99.0'; },
  ]) {
    const copy = structuredClone(data); mutate(copy);
    assert.throws(() => validate(copy, manifest));
  }
});

test('unknown, malformed, and incomplete tokens fail with file context', () => {
  for (const token of ['%%release.missing%%', '%%release.CORE%%', '%%release.core%', '%release.core%%', '%%release.core', '%%release.%%']) {
    assert.throws(() => resolve(`First line\n${token}`, values, 'example.mdx'), /example.mdx:2:/);
  }
});

test('preprocessor renders prose, inline/fenced code, links, tables and MDX children', async () => {
  const {default: gfm} = await import('remark-gfm');
  const content = `Current %%release.core%%.\n\n\`core@v%%release.core%%\`\n\n\`\`\`bash\nsima-cli neat install core@v%%release.core%%\n\`\`\`\n\n[Apps](https://example.com/v%%release.apps%%)\n\n| SDK |\n| --- |\n| %%release.sdk%% |\n\n<ShellCommand>\nsima-cli neat install sdk@%%release.sdk_channel%%\n</ShellCommand>`;
  const compiled = await evaluate(preprocessor(values)({fileContent: content, filePath: 'fixture.mdx'}), {...runtime, remarkPlugins: [gfm]});
  const html = renderToStaticMarkup(React.createElement(compiled.default, {components: {
    ShellCommand: ({children}) => React.createElement('pre', {'data-command': true}, children),
  }}));
  for (const expected of [`Current ${values.core}`, `core@v${values.core}`, `https://example.com/v${values.apps}`, `<td>${values.sdk}</td>`, `sima-cli neat install sdk@${values.sdk_channel}`]) assert.ok(html.includes(expected), expected);
  assert.ok(!html.includes('%%release'));
});

test('release-only changes invalidate preprocessing and update every locale without source/hash edits', () => {
  const next = {...values, core: '9.8.7', sdk: '9.8.7.6', sdk_channel: 'release-9.8', platform: '9.8.7', apps: '9.8.6', model_compiler: '9.8.5', insight: '9.8.4'};
  assert.notEqual(preprocessor(values).toString(), preprocessor(next).toString());
  for (const locale of ['en', 'ko', 'ja', 'zh-Hant', 'uk']) {
    const base = locale === 'en' ? 'docs' : `website/i18n/${locale}/docusaurus-plugin-content-docs/current`;
    for (const [relative, expected] of [
      ['neat-library/pcie-host.mdx', 'core/pciehost/ubuntu24/amd64@v9.8.7'],
      ['dev-environment/install-the-environment.md', 'sdk@release-9.8'],
      ['dev-environment/offline-installation.md', 'sdk@v9.8.7.6'],
      ['dev-environment/install-model-compiler.md', 'model-compiler/amd64@v9.8.5'],
    ]) {
      const file = `${base}/getting-started/${relative}`;
      const rendered = resolve(fs.readFileSync(path.join(root, file), 'utf8'), next, file);
      assert.ok(rendered.includes(expected), `${file}: ${expected}`);
      assert.ok(!rendered.includes('%%release'));
    }
    for (const name of ['index.md', 'install-the-environment.md']) {
      const source = fs.readFileSync(path.join(root, base, 'getting-started/dev-environment', name), 'utf8');
      const rendered = resolve(source, next);
      assert.ok(source.includes('%%release.sdk_series%%'), `${locale}/${name}: missing series token`);
      assert.ok(rendered.includes(next.sdk_series));
      assert.ok(!/(?<![\d.])2\.1(?!\.\d|\d)/.test(rendered), `${locale}/${name}: stale SDK series`);
    }
    const compatibility = fs.readFileSync(path.join(root, base, 'getting-started/compatibility.mdx'), 'utf8');
    assert.ok(!compatibility.includes('%%release'));
    assert.equal(resolve(compatibility, next), compatibility, 'manual compatibility values stay unchanged');
  }
});

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'release-docs-'));
  t.after(() => fs.rmSync(dir, {recursive: true, force: true}));
  const put = (file, text) => { fs.mkdirSync(path.dirname(path.join(dir, file)), {recursive: true}); fs.writeFileSync(path.join(dir, file), text); };
  put('website/release-data/current.json', JSON.stringify(data));
  put('deps/manifest.json', JSON.stringify(manifest));
  put('sima-i18n.config.json', JSON.stringify({locales: {ja: {}}, canonicalTerms: Object.keys(values).map(key => `%%release.${key}%%`)}));
  put('website/release-data/policy.json', JSON.stringify({rollingPaths: ['docs/example.md'], literalExceptions: [], readmes: [{path: 'README.md', template: 'install core@v%%release.core%%'}]}));
  put('docs/example.md', 'Install %%release.core%%');
  put('website/i18n/ja/docusaurus-plugin-content-docs/current/example.md', 'インストール %%release.core%%');
  put('README.md', `<!-- release-command:start -->\n\n\`\`\`bash\ninstall core@v${values.core}\n\`\`\`\n\n<!-- release-command:end -->`);
  return {dir, put};
}

test('validator rejects localized token drift and literal pins even when the release advances', t => {
  const {dir, put} = fixture(t);
  check(dir);
  put('website/i18n/ja/docusaurus-plugin-content-docs/current/example.md', '%%release.sdk%%');
  assert.throws(() => check(dir), /release-token set differs/);
  put('website/i18n/ja/docusaurus-plugin-content-docs/current/example.md', '%%release.core%%');
  put('docs/example.md', '%%release.core%% and old literal 0.4.0');
  assert.throws(() => check(dir), /literal release pin/);
});

test('README regeneration is explicit, concrete, and idempotent; checks never silently write', t => {
  const {dir, put} = fixture(t);
  put('README.md', '<!-- release-command:start -->\nstale\n<!-- release-command:end -->');
  assert.throws(() => check(dir), /release command is stale/);
  check(dir, {writeReadmes: true});
  const first = fs.readFileSync(path.join(dir, 'README.md'), 'utf8');
  check(dir, {writeReadmes: true});
  assert.equal(fs.readFileSync(path.join(dir, 'README.md'), 'utf8'), first);
  assert.ok(first.includes(`core@v${values.core}`));
  assert.ok(!first.includes('%%release'));
});

test('historical tokens and unused or broadened literal exceptions are rejected', t => {
  const {dir, put} = fixture(t);
  put('docs/release-notes/history.md', 'Historical %%release.core%%');
  assert.throws(() => check(dir), /release history must remain pinned/);
  put('docs/release-notes/history.md', 'Historical 0.1.0');
  const policyFile = path.join(dir, 'website/release-data/policy.json');
  const policy = JSON.parse(fs.readFileSync(policyFile));
  policy.literalExceptions.push({path: 'docs/example.md', line: 'Minimum 0.1.0', count: 1, reason: 'Historical minimum'});
  put('website/release-data/policy.json', JSON.stringify(policy));
  assert.throws(() => check(dir), /Unused literal exception/);
  put('docs/example.md', '%%release.core%%\nMinimum 0.1.0');
  check(dir);
  put('docs/example.md', '%%release.core%%\nMinimum 0.1.0\nMinimum 0.1.0');
  assert.throws(() => check(dir), /literal release pin/);
});

test('rendered output checker rejects unresolved tokens and stale commands', t => {
  const {checkRendered} = require('../scripts/check-rendered-releases.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'release-html-'));
  t.after(() => fs.rmSync(dir, {recursive: true, force: true}));
  const examples = [
    ['getting-started/neat-library/pcie-host', `sima-cli neat install core/pciehost/ubuntu24/amd64@v${values.core}`],
    ['getting-started/dev-environment/install-the-environment', `sima-cli neat install sdk@${values.sdk_channel}`],
    ['getting-started/dev-environment/offline-installation', `sima-cli neat install sdk@v${values.sdk} -t offline-amd64`],
    ['getting-started/dev-environment/install-model-compiler', `sima-cli neat install model-compiler/amd64@v${values.model_compiler}`],
  ];
  for (const [route, expected] of examples) {
    const file = path.join(dir, route + '.html');
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, `<pre>${expected}</pre>`);
  }
  checkRendered(dir, ['en']);
  const target = path.join(dir, examples[0][0] + '.html');
  fs.writeFileSync(target, '<pre>%%release.core%%</pre>');
  assert.throws(() => checkRendered(dir, ['en']), /unresolved release token/);
  fs.writeFileSync(target, '<pre>core@v0.0.1</pre>');
  assert.throws(() => checkRendered(dir, ['en']), /expected concrete release example/);
});


test('manually maintained pages reject release tokens in English and translations', t => {
  const {dir, put} = fixture(t);
  const policyFile = path.join(dir, 'website/release-data/policy.json');
  const policy = JSON.parse(fs.readFileSync(policyFile));
  policy.manualPaths = ['docs/compatibility.mdx'];
  put('website/release-data/policy.json', JSON.stringify(policy));
  put('docs/compatibility.mdx', 'Supported pair: 0.5.0 and 2.1.3.1');
  const translated = 'website/i18n/ja/docusaurus-plugin-content-docs/current/compatibility.mdx';
  put(translated, '対応: 0.5.0 / 2.1.3.1');
  check(dir);
  put('docs/compatibility.mdx', 'Supported pair: %%release.core%%');
  assert.throws(() => check(dir), /manually maintained documentation/);
  put('docs/compatibility.mdx', 'Supported pair: 0.5.0 and 2.1.3.1');
  put(translated, '対応: %%release.core%%');
  assert.throws(() => check(dir), /manually maintained documentation/);
});


test('translation validation catches a missing repeated release token', t => {
  const {dir, put} = fixture(t);
  put('docs/example.md', 'Install %%release.core%%; current: %%release.core%%');
  assert.throws(() => check(dir), /release-token count for core differs/);
});
