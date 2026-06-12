import React, {useEffect, useMemo, useState} from 'react';
import {trackDocsEvent} from '@site/src/lib/analytics';
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

function platformFromLabel(label) {
  const lower = String(label || '').toLowerCase();
  if (lower.includes('devkit')) return 'devkit';
  if (lower.includes('elxr') || lower.includes('sdk')) return 'elxr_sdk';
  return null;
}

export function CodeTab({children}) {
  return <>{children}</>;
}

export function CodeTabs({children}) {
  const tabs = useMemo(() => {
    return React.Children.toArray(children)
      .map((child, index) => {
        if (!React.isValidElement(child)) return null;
        const lang = normalizeLang(child.props?.lang);
        if (!lang) return null;
        const label = defaultLabel(lang, child.props?.label);
        return {
          id: `${lang}-${index}-${String(label).toLowerCase().replace(/\s+/g, '-')}`,
          lang,
          label,
          content: child.props?.children,
        };
      })
      .filter(Boolean);
  }, [children]);

  const [activeLang, setActiveLang] = useState('cpp');
  const [activeTabId, setActiveTabId] = useState(null);

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

  const availableLangs = new Set(tabs.map((t) => t.lang));
  const isLanguageMode = availableLangs.size > 1;

  const selectedLang = isLanguageMode
    ? (availableLangs.has(activeLang) ? activeLang : tabs[0].lang)
    : tabs[0].lang;

  const selectedTab = isLanguageMode
    ? (tabs.find((t) => t.lang === selectedLang) || tabs[0])
    : (tabs.find((t) => t.id === activeTabId) || tabs[0]);

  return (
    <div className={styles.root} data-neat-language-aware={isLanguageMode ? 'true' : undefined}>
      <div
        className={styles.tabList}
        role="tablist"
        aria-label={isLanguageMode ? 'Language tabs' : 'Code tabs'}>
        {tabs.map((tab) => {
          const active = isLanguageMode ? tab.lang === selectedLang : tab.id === selectedTab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={`${styles.tabButton} ${active ? styles.tabButtonActive : ''}`}
              onClick={() => {
                if (isLanguageMode) {
                  setPreferredLang(tab.lang);
                  setActiveLang(tab.lang);
                  trackDocsEvent('language_tab_select', {
                    language: tab.lang,
                    tab_label: tab.label,
                  });
                  return;
                }
                setActiveTabId(tab.id);
                const platform = platformFromLabel(tab.label);
                if (platform) {
                  trackDocsEvent('platform_select', {
                    platform,
                    tab_label: tab.label,
                  });
                }
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
