import React from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

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
  const buildInfo = siteConfig.customFields?.buildInfo;

  if (!buildInfo?.showBanner) {
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
            <stop offset="58%" stopColor="#1f6feb" />
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

export default function Root({children}) {
  return (
    <>
      <BuildInfoBanner />
      {children}
    </>
  );
}
