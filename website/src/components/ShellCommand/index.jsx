import React, {useEffect, useRef, useState} from 'react';
import IconCopy from '@theme/Icon/Copy';
import IconSuccess from '@theme/Icon/Success';
import styles from './styles.module.css';

// Canonical execution environments. This list is closed on purpose: an unknown
// value throws during the docs build so the vocabulary cannot drift again.
// Combine tokens with "|" when a command runs in either place, for example
// prompt="sdk|devkit".
const ENVIRONMENT_CLASSES = {
  host: styles.envHost,
  sdk: styles.envSdk,
  devkit: styles.envDevkit,
  'pcie-host': styles.envPcieHost,
};

const ENVIRONMENTS = Object.keys(ENVIRONMENT_CLASSES);

function parseEnvironments(prompt) {
  const tokens = String(prompt)
    .split('|')
    .map((token) => token.trim())
    .filter(Boolean);

  if (!tokens.length) {
    throw new Error(
      `<ShellCommand prompt="${prompt}"> has no environment. ` +
        `Use one of: ${ENVIRONMENTS.join(', ')}.`,
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

function promptLabel(tokens) {
  if (tokens.length === 1) return tokens[0];
  return `${tokens.slice(0, -1).join(', ')} or ${tokens[tokens.length - 1]}`;
}

function textFromNode(node) {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textFromNode).join('');
  if (React.isValidElement(node)) return textFromNode(node.props?.children);
  return '';
}

function normalizeCommand(children) {
  const raw = textFromNode(React.Children.toArray(children)).replace(/\r\n/g, '\n').trim();
  const lines = raw.split('\n');
  const indents = lines
    .filter((line) => line.trim())
    .map((line) => line.match(/^\s*/)?.[0].length ?? 0);
  const minIndent = indents.length ? Math.min(...indents) : 0;
  return lines.map((line) => line.slice(minIndent)).join('\n');
}

export default function ShellCommand({
  children,
  prompt = 'host',
  note,
  cwd,
}) {
  const [copied, setCopied] = useState(false);
  const copyTimeout = useRef(undefined);
  const command = normalizeCommand(children);
  const lines = command.split('\n');
  const environments = parseEnvironments(prompt);
  const label = promptLabel(environments);
  const environmentClass =
    environments.length === 1 ? ENVIRONMENT_CLASSES[environments[0]] : styles.envMulti;
  const caption = [note, cwd ? `run from ${cwd}` : ''].filter(Boolean).join(' · ');

  useEffect(() => () => window.clearTimeout(copyTimeout.current), []);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.clearTimeout(copyTimeout.current);
      copyTimeout.current = window.setTimeout(() => setCopied(false), 1000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className={`${styles.shellCommand} ${environmentClass}`}>
      {caption ? <div className={styles.caption}>{caption}</div> : null}
      <div className={styles.terminal}>
        <button
          type="button"
          className={`${styles.copyButton} ${copied ? styles.copyButtonCopied : ''}`}
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy command'}
          title="Copy">
          <span className={styles.copyButtonIcons} aria-hidden="true">
            <IconCopy className={styles.copyButtonIcon} />
            <IconSuccess className={styles.copyButtonSuccessIcon} />
          </span>
        </button>
        <pre className={styles.pre}>
          <code className={styles.code}>
            {lines.map((line, index) => (
              <span className={styles.line} key={`${index}-${line}`}>
                <span className={styles.prompt} aria-hidden="true">
                  {label}$
                </span>
                <span className={styles.command}>{line || ' '}</span>
              </span>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
