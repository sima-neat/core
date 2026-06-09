// Docusaurus' DocSidebarItem/Category injects an SSR-fallback `href` on
// every collapsible category that has no explicit link, pointing it at the
// first child page (see node_modules/@docusaurus/theme-classic/.../Category
// useCategoryHrefWithSSRFallback). The fallback is meant to keep the link
// reachable for screen readers and no-JS clients, but it also means a click
// before React hydration triggers a real navigation instead of just toggling
// the section. We strip the href on category anchors that carry the
// `menu__link--sublist-caret` class — that class is only added when the
// category has no real link, so the href can't be anything but the fallback.

function stripFallbackHrefs(root) {
  const carets = (root || document).querySelectorAll(
    'a.menu__link--sublist-caret[href]',
  );
  carets.forEach((anchor) => anchor.removeAttribute('href'));
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => stripFallbackHrefs(), {once: true});
  } else {
    stripFallbackHrefs();
  }

  // Re-run on every navigation: Docusaurus re-renders the sidebar with
  // fresh SSR-style HTML on client-side route changes.
  const observer = new MutationObserver(() => stripFallbackHrefs());
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
}
