/** The initial shell only constructs a local login URL. Account validates the
 * requested continuation with accountReturnHref before displaying or following it.
 */
export const personalAISettingsHref = '#/account/ai';

export function accountEntryHref(destination: string): string {
  try {
    const raw = destination.replace(/^#\/?/, ""), query = raw.indexOf('?');
    const route = destination.startsWith('#')
      ? query < 0 ? decodeURIComponent(raw) : decodeURIComponent(raw.slice(0, query)) + raw.slice(query)
      : destination;
    if (route === 'account/ai') return personalAISettingsHref;
    if (/^(?:[a-z][\w+.-]*:|[\/#]|account(?:\/|$))/i.test(route) || /[\r\n\\]/.test(route)) return '#/account';
    // A sharing secret stays in this tab; only the static join page crosses sign-in.
    if (/^rebuttal\/join\/[A-Za-z0-9_-]{43}$/.test(route)) return "#/account/continue/rebuttal%2Fjoin";
    return `#/account/continue/${encodeURIComponent(route)}`;
  } catch { return '#/account'; }
}

export function isAccountRoute(route: string): boolean {
  return route === 'account' || route === 'account/ai' || route.startsWith('account/continue/') || route.startsWith('account/error/');
}
