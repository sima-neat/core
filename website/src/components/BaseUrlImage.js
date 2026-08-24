import React from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function BaseUrlImage({
  asObject = false,
  asobject = false,
  src,
  alt,
  children: _children,
  ...props
}) {
  const resolvedSrc = useBaseUrl(src);

  if (asObject || asobject) {
    return (
      <object
        {...props}
        type="image/svg+xml"
        data={resolvedSrc}
        aria-label={alt}
      >
        <img src={resolvedSrc} alt={alt} loading="lazy" />
      </object>
    );
  }

  return <img {...props} src={resolvedSrc} alt={alt} />;
}
