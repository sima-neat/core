import React from 'react';
import MDXComponents from '@theme-original/MDXComponents';
import CodePopup from '@site/src/components/CodePopup';
import ApiReferenceLink from '@site/src/components/ApiReferenceLink';
import LanguageContent from '@site/src/components/LanguageContent';
import CodeTabs, {CodeTab} from '@site/src/components/CodeTabs';
import ShellCommand from '@site/src/components/ShellCommand';
import OperatorTable from '@site/src/components/OperatorTable';
import BaseUrlImage from '@site/src/components/BaseUrlImage';
import BaseUrlLink from '@site/src/components/BaseUrlLink';

export default {
  ...MDXComponents,
  CodePopup,
  LanguageContent,
  languagecontent: LanguageContent,
  CodeTabs,
  CodeTab,
  ShellCommand,
  OperatorTable,
  BaseUrlImage,
  BaseUrlLink,
  codetabs: CodeTabs,
  codetab: CodeTab,
  shellcommand: ShellCommand,
  operatortable: OperatorTable,
  baseurlimage: BaseUrlImage,
  baseurllink: BaseUrlLink,
  a: ApiReferenceLink,
};
