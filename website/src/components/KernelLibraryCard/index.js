import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

export default function KernelLibraryCard({title, children}) {
  const {siteConfig} = useDocusaurusContext();
  if (!siteConfig.customFields.kernelLibrary) return null;
  return <li><Link className="overview-link-card" to="/kernel-library/">
    <strong>{title}</strong><span>{children}</span>
  </Link></li>;
}
