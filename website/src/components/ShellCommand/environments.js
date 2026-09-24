/**
 * Canonical execution environments for command blocks.
 *
 * The list is closed on purpose: an unknown value throws during the docs build,
 * so the prompt vocabulary cannot drift back into the eleven ad-hoc spellings it
 * had before. Combine tokens with "|" when a command runs in either place, for
 * example prompt="sdk|devkit".
 *
 * Kept free of React and CSS imports so the validation can be unit tested
 * directly -- it is build-critical, and a regression here breaks the docs build.
 */

export const ENVIRONMENTS = ['host', 'sdk', 'devkit', 'pcie-host'];

export function parseEnvironments(prompt) {
  // Split without discarding empties: a stray separator such as "sdk|" or
  // "sdk||devkit" is malformed, and silently normalising it away would defeat
  // the point of validating here at all.
  const tokens = String(prompt)
    .split('|')
    .map((token) => token.trim());

  if (tokens.every((token) => !token)) {
    throw new Error(
      `<ShellCommand prompt="${prompt}"> has no environment. ` +
        `Use one of: ${ENVIRONMENTS.join(', ')}.`,
    );
  }

  if (tokens.some((token) => !token)) {
    throw new Error(
      `<ShellCommand prompt="${prompt}"> has an empty segment around a "|". ` +
        `Separate exactly two or more environments, for example prompt="sdk|devkit".`,
    );
  }

  const unknown = tokens.filter((token) => !ENVIRONMENTS.includes(token));
  if (unknown.length) {
    throw new Error(
      `<ShellCommand prompt="${prompt}"> uses unknown environment(s): ` +
        `${unknown.join(', ')}. Valid environments are: ${ENVIRONMENTS.join(', ')}. ` +
        'Combine them with "|" (for example prompt="sdk|devkit") when a command ' +
        'runs in either place.',
    );
  }

  return tokens;
}

export function promptLabel(tokens) {
  if (tokens.length === 1) return tokens[0];
  return `${tokens.slice(0, -1).join(', ')} or ${tokens[tokens.length - 1]}`;
}
