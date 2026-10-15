/** Fetch a fresh document while preserving the page the user was opening. */
export function recoveryUrl(href: string, now = Date.now()): string {
  const url = new URL(href);
  url.searchParams.set('_paperleet_reload', String(now));
  return url.href;
}

export function reloadApp() {
  window.location.replace(recoveryUrl(window.location.href));
}

export function finishAppStartup() {
  document.dispatchEvent(new Event('paperleet:ready'));
  const url = new URL(window.location.href);
  if (url.searchParams.has('_paperleet_reload')) {
    url.searchParams.delete('_paperleet_reload');
    window.history.replaceState(window.history.state, '', url.href);
  }
}
