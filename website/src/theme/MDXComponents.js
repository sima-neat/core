import React from 'react';
import MDXComponents from '@theme-original/MDXComponents';
import CodePopup from '@site/src/components/CodePopup';
import ApiReferenceLink from '@site/src/components/ApiReferenceLink';

export default {
  ...MDXComponents,
  CodePopup,
  a: ApiReferenceLink,
};
