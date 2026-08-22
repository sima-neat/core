import React from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function BaseUrlLink({href, children, ...props}) {
  return (
    <a {...props} href={useBaseUrl(href)}>
      {children}
    </a>
  );
}
