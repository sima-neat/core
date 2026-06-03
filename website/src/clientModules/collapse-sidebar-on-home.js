// Force a full page reload when the user clicks any link back to the docs
// home page. A fresh load resets the sidebar to its configured collapsed
// state, which (paired with themeConfig.docs.sidebar.autoCollapseCategories)
// gives the TOC a clean accordion reset without having to manually click
// open category toggles.

function isHomeHref(href) {
  if (!href) return false;
  try {
    const url = new URL(href, window.location.origin);
    const path = url.pathname.replace(/\/+$/, '') || '/';
    return path === '/';
  } catch (_) {
    return false;
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener(
    'click',
    (event) => {
      // Allow modified clicks (open-in-new-tab, middle-click, etc.) through.
      if (
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }
      const link = event.target.closest('a');
      if (!link) return;
      if (!isHomeHref(link.getAttribute('href'))) return;
      event.preventDefault();
      event.stopPropagation();
      const here = window.location.pathname.replace(/\/+$/, '') || '/';
      if (here === '/') {
        // Same URL: assigning href is a no-op, so force a reload.
        window.location.reload();
      } else {
        window.location.href = link.href;
      }
    },
    true, // capture phase so we run before React Router's handler
  );
}
