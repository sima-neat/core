import assert from 'node:assert/strict';
import test from 'node:test';

import {ENVIRONMENTS, parseEnvironments, promptLabel} from './environments.js';

test('every canonical environment is accepted on its own', () => {
  for (const env of ENVIRONMENTS) {
    assert.deepEqual(parseEnvironments(env), [env]);
    assert.equal(promptLabel([env]), env);
  }
});

test('environments combine with "|" and read as prose', () => {
  assert.deepEqual(parseEnvironments('sdk|devkit'), ['sdk', 'devkit']);
  assert.equal(promptLabel(['sdk', 'devkit']), 'sdk or devkit');

  assert.deepEqual(parseEnvironments('sdk|host'), ['sdk', 'host']);
  assert.equal(promptLabel(['sdk', 'host']), 'sdk or host');

  assert.deepEqual(parseEnvironments('sdk|devkit|pcie-host'), ['sdk', 'devkit', 'pcie-host']);
  assert.equal(promptLabel(['sdk', 'devkit', 'pcie-host']), 'sdk, devkit or pcie-host');
});

test('surrounding whitespace is tolerated', () => {
  assert.deepEqual(parseEnvironments(' sdk | devkit '), ['sdk', 'devkit']);
});

test('the legacy spellings this vocabulary replaced are rejected', () => {
  // If any of these is ever accepted again, the drift has returned.
  const legacy = [
    'user-host-machine',
    'username@neat-sdk-latest',
    'username@ros2-sdk-latest',
    'sdk-or-devkit',
    'sdk-or-pcie-host',
    'sdk-devkit-or-pcie-host',
    'modalix',
    'online-machine',
    'offline-host',
  ];
  for (const value of legacy) {
    assert.throws(() => parseEnvironments(value), /unknown environment/, `accepted "${value}"`);
  }
});

test('an empty prompt is rejected', () => {
  for (const value of ['', '   ', '|', ' | ']) {
    assert.throws(() => parseEnvironments(value), /has no environment/, `accepted "${value}"`);
  }
});

test('one bad token rejects the whole combination', () => {
  assert.throws(() => parseEnvironments('sdk|nope'), /unknown environment\(s\): nope/);
});

test('rejection messages name the offending value and the valid set', () => {
  try {
    parseEnvironments('modalix');
    assert.fail('expected a throw');
  } catch (error) {
    assert.match(error.message, /prompt="modalix"/);
    for (const env of ENVIRONMENTS) assert.ok(error.message.includes(env), `message omits ${env}`);
  }
});
