import React from 'react';
import MDXComponents from '@theme-original/MDXComponents';
import CodePopup from '@site/src/components/CodePopup';
import ApiReferenceLink from '@site/src/components/ApiReferenceLink';
import LanguageContent from '@site/src/components/LanguageContent';
import CodeTabs, {CodeTab} from '@site/src/components/CodeTabs';

export default {
  ...MDXComponents,
  CodePopup,
  LanguageContent,
  languagecontent: LanguageContent,
  CodeTabs,
  CodeTab,
  codetabs: CodeTabs,
  codetab: CodeTab,
  a: ApiReferenceLink,
};
