import React from 'react';
import {useLocation} from '@docusaurus/router';
import {HtmlClassNameProvider} from '@docusaurus/theme-common';
import {DocProvider} from '@docusaurus/plugin-content-docs/client';
import DocItemMetadata from '@theme/DocItem/Metadata';
import DocItemLayout from '@theme/DocItem/Layout';

function MissingDocContent() {
  const {pathname} = useLocation();
  return (
    <DocItemLayout>
      <article className="markdown">
        <h1>Reference page unavailable</h1>
        <p>
          Docusaurus did not provide the compiled doc payload for
          <code>{` ${pathname} `}</code>
          during static rendering.
        </p>
      </article>
    </DocItemLayout>
  );
}

export default function DocItem(props) {
  const content = props?.content;

  if (!content?.metadata) {
    return <MissingDocContent />;
  }

  const docHtmlClassName = `docs-doc-id-${content.metadata.id}`;
  const MDXComponent = content;

  return (
    <DocProvider content={content}>
      <HtmlClassNameProvider className={docHtmlClassName}>
        <DocItemMetadata />
        <DocItemLayout>
          <MDXComponent />
        </DocItemLayout>
      </HtmlClassNameProvider>
    </DocProvider>
  );
}
