import React, {useEffect, useMemo, useState} from 'react';

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

export default function LanguageContent({lang, children}) {
  const targetLang = useMemo(() => normalizeLang(lang), [lang]);
  const [activeLang, setActiveLang] = useState('cpp');

  useEffect(() => {
    const sync = () => setActiveLang(getPreferredLang());
    sync();
    window.addEventListener('neat:langchange', sync);
    window.addEventListener('storage', sync);
    return () => {
      window.removeEventListener('neat:langchange', sync);
      window.removeEventListener('storage', sync);
    };
  }, []);

  if (!targetLang || targetLang !== activeLang) {
    return null;
  }

  return (
    <>
      <span hidden data-neat-language-aware="true" />
      {children}
    </>
  );
}
