import React from 'react';
import {useLocation} from '@docusaurus/router';
import {HtmlClassNameProvider} from '@docusaurus/theme-common';
import {DocProvider} from '@docusaurus/plugin-content-docs/client';
import Layout from '@theme/Layout';
import DocItemMetadata from '@theme/DocItem/Metadata';
import DocItemLayout from '@theme/DocItem/Layout';

// Rendered when Docusaurus hands us a doc route without a compiled payload
// (seen transiently in `docusaurus start` dev). It must NOT use DocItemLayout
// or any DocItem child: those call `useDoc()`, which requires a DocProvider we
// have no content to populate — rendering them here throws "useDoc is called
// outside the <DocProvider>". Use the plain site Layout instead.
function MissingDocContent() {
  const {pathname} = useLocation();
  return (
    <Layout>
      <main className="container margin-vert--lg">
        <article className="markdown">
          <h1>Reference page unavailable</h1>
          <p>
            The compiled doc payload for
            <code>{` ${pathname} `}</code>
            failed to load. In the dev server this is usually a stale page chunk
            after a rebuild — reload the page (a hard refresh, Cmd/Ctrl+Shift+R)
            to fetch the current build.
          </p>
        </article>
      </main>
    </Layout>
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
