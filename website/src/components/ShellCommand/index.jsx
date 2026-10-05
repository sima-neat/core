import React, {useEffect, useRef, useState} from 'react';
import IconCopy from '@theme/Icon/Copy';
import IconSuccess from '@theme/Icon/Success';
import styles from './styles.module.css';
import {parseEnvironments, promptLabel} from './environments';

// Environment tokens and their validation live in environments.js so they can
// be unit tested without React or CSS. Only the colour mapping needs `styles`.
const ENVIRONMENT_CLASSES = {
  host: styles.envHost,
  sdk: styles.envSdk,
  devkit: styles.envDevkit,
  'pcie-host': styles.envPcieHost,
};

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

export default function ShellCommand({children, prompt = 'host'}) {
  const [copied, setCopied] = useState(false);
  const copyTimeout = useRef(undefined);
  const command = normalizeCommand(children);
  const lines = command.split('\n');
  const environments = parseEnvironments(prompt);
  const label = promptLabel(environments);
  const environmentClass =
    environments.length === 1 ? ENVIRONMENT_CLASSES[environments[0]] : styles.envMulti;

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
                <span
                  className={styles.prompt}
                  aria-hidden={index > 0 ? 'true' : undefined}>
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
