import React, {useMemo, useState} from 'react';
import styles from './styles.module.css';

const YES = '✅';
const NO = '❌';

// Columns whose values are support flags, so they get the tri-state filter.
const FLAG_COLUMNS = [
  {key: 'int8', label: 'INT8'},
  {key: 'bf16', label: 'BF16'},
  {key: 'fived', label: '5D'},
];

function onnxDocUrl(name) {
  return `https://onnx.ai/onnx/operators/onnx__${name}.html`;
}

function textFromNode(node) {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textFromNode).join('');
  if (React.isValidElement(node)) return textFromNode(node.props?.children);
  return '';
}

/**
 * Parse the operator rows. The data is authored as a JSON array, passed either
 * as the `operators` attribute or as the element's children — children is what
 * the generator uses, since a 60-row payload does not belong in an attribute.
 */
function parseOperators(operators, children) {
  const raw = (operators ?? textFromNode(children)).trim();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function OperatorTable({operators, children}) {
  const rows = useMemo(() => parseOperators(operators, children), [operators, children]);
  const [query, setQuery] = useState('');
  // Each flag column filters independently: '' = any, '✅' = supported only.
  const [flags, setFlags] = useState({});
  const [sort, setSort] = useState({key: 'name', dir: 1});

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = rows.filter((row) => {
      for (const {key} of FLAG_COLUMNS) {
        if (flags[key] && row[key] !== flags[key]) return false;
      }
      if (!needle) return true;
      // Search covers the constraint text too, so "ceil" or "channel axis"
      // finds the operator even though constraints render below the table.
      return `${row.name} ${row.constraint ?? ''}`.toLowerCase().includes(needle);
    });
    const {key, dir} = sort;
    return [...filtered].sort((a, b) => {
      const av = a[key] ?? '';
      const bv = b[key] ?? '';
      if (key === 'opset') {
        const an = parseInt(av, 10);
        const bn = parseInt(bv, 10);
        if (Number.isNaN(an) && Number.isNaN(bn)) return 0;
        if (Number.isNaN(an)) return 1;
        if (Number.isNaN(bn)) return -1;
        return (an - bn) * dir;
      }
      return String(av).localeCompare(String(bv)) * dir;
    });
  }, [rows, query, flags, sort]);

  if (!rows.length) return null;

  function toggleSort(key) {
    setSort((prev) => (prev.key === key ? {key, dir: -prev.dir} : {key, dir: 1}));
  }

  function cycleFlag(key) {
    setFlags((prev) => {
      const current = prev[key] ?? '';
      const next = current === '' ? YES : current === YES ? NO : '';
      return {...prev, [key]: next};
    });
  }

  function sortIndicator(key) {
    if (sort.key !== key) return '';
    return sort.dir === 1 ? ' ▲' : ' ▼';
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.controls}>
        <input
          type="search"
          className={styles.search}
          placeholder="Search operators and constraints…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label="Search operators and constraints"
        />
        <div className={styles.filters}>
          {FLAG_COLUMNS.map(({key, label}) => {
            const state = flags[key] ?? '';
            return (
              <button
                key={key}
                type="button"
                className={state ? styles.filterOn : styles.filter}
                onClick={() => cycleFlag(key)}
                aria-pressed={Boolean(state)}
              >
                {label} {state || 'any'}
              </button>
            );
          })}
        </div>
      </div>

      <div className={styles.count} role="status" aria-live="polite">
        {visible.length} of {rows.length} operators
      </div>

      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>
                <button type="button" className={styles.sort} onClick={() => toggleSort('name')}>
                  Operator{sortIndicator('name')}
                </button>
              </th>
              {FLAG_COLUMNS.map(({key, label}) => (
                <th key={key} className={styles.center}>
                  <button type="button" className={styles.sort} onClick={() => toggleSort(key)}>
                    {label}
                    {sortIndicator(key)}
                  </button>
                </th>
              ))}
              <th className={styles.center}>
                <button type="button" className={styles.sort} onClick={() => toggleSort('opset')}>
                  Opset{sortIndicator('opset')}
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((row) => (
              <tr key={row.name}>
                <td>
                  <a href={onnxDocUrl(row.name)} target="_blank" rel="noopener noreferrer">
                    <code>{row.name}</code>
                  </a>
                </td>
                {FLAG_COLUMNS.map(({key}) => (
                  <td key={key} className={styles.center}>
                    {row[key]}
                  </td>
                ))}
                <td className={styles.center}>{row.opset}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {visible.length === 0 && (
        <p className={styles.empty}>No operators match those filters.</p>
      )}
    </div>
  );
}
