function normalizeDeveloperCenterBrand() {
  document.querySelectorAll('a.navbar__brand').forEach((brandLink) => {
    brandLink.removeAttribute('target');
    brandLink.removeAttribute('rel');
  });
}

if (typeof document !== 'undefined') {
  normalizeDeveloperCenterBrand();

  const observer = new MutationObserver(normalizeDeveloperCenterBrand);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
}
