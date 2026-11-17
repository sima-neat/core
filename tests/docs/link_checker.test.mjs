// Run via npm exec so this verifies the same pinned dependency as the shell gate:
// npm exec --yes --package=linkinator@8.1.0 -- sh -c \
//   'LINKINATOR_CLI="$(command -v linkinator)" node --test tests/docs/link_checker.test.mjs'
import assert from 'node:assert/strict';
import { realpathSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';

const moduleDir = dirname(realpathSync(process.env.LINKINATOR_CLI));
const { LinkChecker } = await import(pathToFileURL(join(moduleDir, 'index.js')));

test('an early retry timer keeps the queue alive until the retry executes', () => {
  // Force the first wakeup to happen before its deadline. The old queue exits
  // with code 13 because only its unreferenced fallback interval remains.
  const source = `
    import { Queue } from ${JSON.stringify(pathToFileURL(join(moduleDir, 'queue.js')).href)};
    const original = globalThis.setTimeout;
    let first = true;
    globalThis.setTimeout = (callback, delay, ...args) => {
      const early = first;
      first = false;
      return original(callback, early ? 0 : delay, ...args);
    };
    const queue = new Queue({ concurrency: 1 });
    queue.add(async () => console.log('retry executed'), { delay: 100.5 });
    await queue.onIdle();
  `;
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', source], {
    encoding: 'utf8', timeout: 5000,
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /retry executed/);
});

for (const permanent of [false, true]) {
  test(permanent ? 'a genuine 404 fails the gate' : 'a transient 503 is retried successfully', async () => {
    let requests = 0;
    const server = createServer((_req, res) => {
      requests++;
      res.writeHead(permanent ? 404 : requests === 1 ? 503 : 200, {
        'Content-Type': 'text/html',
      });
      res.end('<html><body>fixture</body></html>');
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    try {
      const result = await new LinkChecker().check({
        path: `http://127.0.0.1:${server.address().port}/`, recurse: true,
        retryErrors: true, retryErrorsCount: 1, retryErrorsJitter: 0,
        timeout: 2000, concurrency: 2,
      });
      assert.equal(result.passed, !permanent);
      assert.equal(requests, permanent ? 1 : 2);
      assert.equal(result.links[0].status, permanent ? 404 : 200);
    } finally {
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
    }
  });
}
