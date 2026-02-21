import React, {useEffect, useMemo, useState} from 'react';
import styles from './styles.module.css';

const LANG_PREF_KEY = 'neat-docs-language';

function normalizeLang(lang) {
  if (!lang) return null;
  const lower = String(lang).toLowerCase();
  if (lower === 'python' || lower === 'py') return 'py';
  if (lower === 'c++' || lower === 'cpp' || lower === 'cxx' || lower === 'cc') return 'cpp';
  return lower;
}

function getPreferredLang() {
  if (typeof window === 'undefined') return 'cpp';
  return normalizeLang(window.localStorage.getItem(LANG_PREF_KEY)) || 'cpp';
}

function setPreferredLang(lang) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(LANG_PREF_KEY, lang);
  window.dispatchEvent(new CustomEvent('neat:langchange', {detail: {lang}}));
}

function defaultLabel(lang, fallback) {
  if (fallback) return fallback;
  if (lang === 'py') return 'Python';
  if (lang === 'cpp') return 'C++';
  return String(lang || 'Code');
}

export function CodeTab({children}) {
  return <>{children}</>;
}

export function CodeTabs({children}) {
  const tabs = useMemo(() => {
    return React.Children.toArray(children)
      .map((child) => {
        if (!React.isValidElement(child)) return null;
        const lang = normalizeLang(child.props?.lang);
        if (!lang) return null;
        return {
          lang,
          label: defaultLabel(lang, child.props?.label),
          content: child.props?.children,
        };
      })
      .filter(Boolean);
  }, [children]);

  const [activeLang, setActiveLang] = useState(getPreferredLang);

  useEffect(() => {
    const sync = (event) => setActiveLang(normalizeLang(event?.detail?.lang) || getPreferredLang());
    sync();
    window.addEventListener('neat:langchange', sync);
    window.addEventListener('storage', sync);
    return () => {
      window.removeEventListener('neat:langchange', sync);
      window.removeEventListener('storage', sync);
    };
  }, []);

  if (!tabs.length) return null;

  const available = new Set(tabs.map((t) => t.lang));
  const selectedLang = available.has(activeLang) ? activeLang : tabs[0].lang;
  const selectedTab = tabs.find((t) => t.lang === selectedLang) || tabs[0];

  return (
    <div className={styles.root}>
      <div className={styles.tabList} role="tablist" aria-label="Language tabs">
        {tabs.map((tab) => {
          const active = tab.lang === selectedLang;
          return (
            <button
              key={tab.lang}
              type="button"
              role="tab"
              aria-selected={active}
              className={`${styles.tabButton} ${active ? styles.tabButtonActive : ''}`}
              onClick={() => {
                setPreferredLang(tab.lang);
                setActiveLang(tab.lang);
              }}>
              {tab.label}
            </button>
          );
        })}
      </div>
      <div role="tabpanel" className={styles.panel}>
        {selectedTab.content}
      </div>
    </div>
  );
}

export default CodeTabs;
