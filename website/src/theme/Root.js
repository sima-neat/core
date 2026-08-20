import React, {useEffect, useState} from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useBaseUrl from "@docusaurus/useBaseUrl";
import {useLocation} from "@docusaurus/router";

const I18N_PREVIEW_PARAM = "i18n";
const I18N_PREVIEW_SESSION_KEY = "neat-docs-i18n-preview";
const DOCS_LOCALE_COOKIE = "sima-neat-locale";
const ENGLISH_ONLY_ROUTE_PREFIXES = [
  "/doxygen",
  "/reference",
];
const LOCALIZED_UI = {
  en: {
    code: "Code",
    gettingStarted: "Getting Started",
    generatedNotice:
      "Generated and reference documentation in this section is currently available in English only.",
    preferredCodeLanguage: "Preferred programming language",
  },
  ko: {
    code: "코드",
    gettingStarted: "시작하기",
    generatedNotice: "이 섹션의 생성 문서와 참조 문서는 현재 영어로만 제공됩니다.",
    preferredCodeLanguage: "선호하는 프로그래밍 언어",
  },
  ja: {
    code: "コード",
    gettingStarted: "はじめに",
    generatedNotice:
      "このセクションの生成ドキュメントとリファレンスは、現在英語版のみ提供しています。",
    preferredCodeLanguage: "優先するプログラミング言語",
  },
  "zh-Hant": {
    code: "程式碼",
    gettingStarted: "開始使用",
    generatedNotice: "本區段自動產生的文件與參考資料目前僅提供英文版。",
    preferredCodeLanguage: "偏好的程式語言",
  },
  uk: {
    code: "Код",
    gettingStarted: "Початок роботи",
    generatedNotice:
      "Згенерована та довідкова документація в цьому розділі наразі доступна лише англійською мовою.",
    preferredCodeLanguage: "Бажана мова програмування",
  },
};

function preferredDocsLocale() {
  if (typeof window === "undefined") return "";
  if (window.__NEAT_DOCS_PREFERRED_LOCALE__) {
    return window.__NEAT_DOCS_PREFERRED_LOCALE__;
  }
  const entry = document.cookie
    .split("; ")
    .find((cookie) => cookie.startsWith(`${DOCS_LOCALE_COOKIE}=`));
  return entry ? decodeURIComponent(entry.split("=").slice(1).join("=")) : "";
}

function localizationEnabled(search, currentLocale) {
  if (new URLSearchParams(search).get(I18N_PREVIEW_PARAM) === "1") return true;
  if (typeof window === "undefined") return false;

  try {
    return (
      window.sessionStorage.getItem(I18N_PREVIEW_SESSION_KEY) === "1" ||
      preferredDocsLocale() === currentLocale
    );
  } catch {
    return false;
  }
}

function I18nPreviewGate({children}) {
  const {i18n, siteConfig} = useDocusaurusContext();
  const location = useLocation();
  const [mounted, setMounted] = useState(false);
  const previewEnabled = localizationEnabled(location.search, i18n.currentLocale);
  const isLocalized = i18n.currentLocale !== i18n.defaultLocale;

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || !isLocalized || previewEnabled) return;

    const baseUrl = siteConfig.baseUrl.endsWith("/")
      ? siteConfig.baseUrl
      : `${siteConfig.baseUrl}/`;
    const localePrefix = `${baseUrl}${i18n.currentLocale}/`;
    const englishPath = location.pathname.startsWith(localePrefix)
      ? `${baseUrl}${location.pathname.slice(localePrefix.length)}`
      : baseUrl;
    window.location.replace(`${englishPath}${location.search}${location.hash}`);
  }, [i18n.currentLocale, isLocalized, location, mounted, previewEnabled, siteConfig.baseUrl]);

  if (mounted && isLocalized && !previewEnabled) return null;
  return children;
}

function LocalizationScopeNotice() {
  const {i18n} = useDocusaurusContext();
  const location = useLocation();
  const previewEnabled = localizationEnabled(location.search, i18n.currentLocale);
  const isEnglishOnlyPrefix = ENGLISH_ONLY_ROUTE_PREFIXES.some(
    (prefix) => location.pathname.includes(`${prefix}/`) || location.pathname.endsWith(prefix),
  );
  const isTutorial = /\/tutorials(?:\/|$)/.test(location.pathname);
  const isLocalizedTutorial = /\/tutorials\/before-you-run\/?$/.test(location.pathname);
  const isEnglishOnlySection = isEnglishOnlyPrefix || (isTutorial && !isLocalizedTutorial);
  const ui = LOCALIZED_UI[i18n.currentLocale] || LOCALIZED_UI.en;

  if (!previewEnabled || !isEnglishOnlySection) return null;

  return (
    <aside className="localization-scope-notice" role="status">
      {ui.generatedNotice}
    </aside>
  );
}

function BannerLink({href, children}) {
  if (!href) {
    return <strong>{children}</strong>;
  }

  return (
    <a
      className="build-info-banner__link"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
    >
      <strong>{children}</strong>
    </a>
  );
}

function BuildInfoBanner() {
  const {siteConfig} = useDocusaurusContext();
  const location = useLocation();
  const buildInfo = siteConfig.customFields?.buildInfo;
  const hideInApiPreview = new URLSearchParams(location.search).get("apiPreview") === "1";
  const [visible, setVisible] = useState(Boolean(buildInfo?.showBanner) && !hideInApiPreview);

  useEffect(() => {
    if (!buildInfo?.showBanner || hideInApiPreview) {
      setVisible(false);
      return undefined;
    }

    setVisible(true);
    const timer = window.setTimeout(() => {
      setVisible(false);
    }, 3000);

    return () => window.clearTimeout(timer);
  }, [buildInfo?.showBanner, hideInApiPreview]);

  if (!buildInfo?.showBanner || hideInApiPreview || !visible) {
    return null;
  }

  return (
    <aside
      className="build-info-banner"
      aria-label="Non-main documentation build details"
    >
      <svg
        className="build-info-banner__shape"
        viewBox="0 0 1000 72"
        preserveAspectRatio="none"
        aria-hidden="true"
        focusable="false"
      >
        <defs>
          <linearGradient
            id="build-info-banner-gradient"
            x1="0"
            x2="1"
            y1="0"
            y2="0"
          >
            <stop offset="0%" stopColor="#10243a" />
            <stop offset="58%" stopColor="#5998dd" />
            <stop offset="100%" stopColor="#2a9c4f" />
          </linearGradient>
        </defs>
        <path
          d="M8 0H992Q1000 0 996 8L958 64Q954 72 944 72H56Q46 72 42 64L4 8Q0 0 8 0Z"
          fill="url(#build-info-banner-gradient)"
        />
        <path
          d="M44 3H956"
          fill="none"
          stroke="rgba(255,255,255,0.36)"
          strokeLinecap="round"
          strokeWidth="1.4"
        />
      </svg>
      <span className="build-info-banner__label">Preview build</span>
      <span className="build-info-banner__item">
        Branch{" "}
        <BannerLink href={buildInfo.branchUrl}>{buildInfo.branch}</BannerLink>
      </span>
      <span className="build-info-banner__item">
        Commit{" "}
        <BannerLink href={buildInfo.commitUrl}>{buildInfo.commit}</BannerLink>
      </span>
      <span className="build-info-banner__item">
        Built <strong>{buildInfo.builtAt}</strong>
      </span>
    </aside>
  );
}

function SoftwareSubnav() {
  const {i18n, siteConfig} = useDocusaurusContext();
  const location = useLocation();
  const ui = LOCALIZED_UI[i18n.currentLocale] || LOCALIZED_UI.en;
  const githubOrgUrl = siteConfig.customFields?.githubOrgUrl || "https://github.com/sima-neat";
  const links = [
    {
      label: ui.gettingStarted,
      href: useBaseUrl("/getting-started/"),
      active: location.pathname.includes("/getting-started"),
    },
    {
      label: "C++ API",
      href: useBaseUrl("/reference/cppapi/"),
      active: location.pathname.includes("/reference/cppapi"),
    },
    {
      label: "Python API",
      href: useBaseUrl("/reference/pythonapi/"),
      active: location.pathname.includes("/reference/pythonapi"),
    },
  ];

  return (
    <nav className="software-subnav" aria-label="Software documentation">
      <div className="software-subnav__inner">
        <div className="software-subnav__links">
          {links.map((link) => (
            <a
              key={link.label}
              className={`software-subnav__link${link.active ? " software-subnav__link--active" : ""}`}
              href={link.href}
            >
              {link.label}
            </a>
          ))}
        </div>
        <div className="software-subnav__controls">
          <div className="language-pref">
            <label htmlFor="language-pref-select-subnav">{ui.code}</label>
            <select
              id="language-pref-select-subnav"
              data-language-pref-select
              aria-label={ui.preferredCodeLanguage}
            >
              <option value="cpp">C++</option>
              <option value="py">Python</option>
            </select>
          </div>
          <a
            className="software-subnav__link software-subnav__github"
            href={githubOrgUrl}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="SiMa Neat on GitHub"
            title="SiMa Neat on GitHub"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
              <path
                fill="currentColor"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.67 7.67 0 0 1 8 3.86c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
              />
            </svg>
          </a>
        </div>
      </div>
    </nav>
  );
}

export default function Root({children}) {
  return (
    <I18nPreviewGate>
      <BuildInfoBanner />
      <SoftwareSubnav />
      <LocalizationScopeNotice />
      {children}
    </I18nPreviewGate>
  );
}
