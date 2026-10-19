import { useEffect, useRef, useState } from 'react';
import { useAccount } from './social';
import { useI18n } from './i18n';

type Turnstile = {
  render(element: HTMLElement, options: Record<string, unknown>): string;
  remove(id: string): void;
};
declare global { interface Window { turnstile?: Turnstile } }
let loading: Promise<Turnstile> | undefined;

function load(): Promise<Turnstile> {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (!loading) loading = new Promise<Turnstile>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
    script.async = true;
    script.onload = () => window.turnstile ? resolve(window.turnstile) : reject(new Error('Security check unavailable'));
    script.onerror = () => reject(new Error('Security check unavailable'));
    document.head.append(script);
  }).catch(error => { loading = undefined; throw error; });
  return loading;
}

function LoginChallenge({ sitekey, onToken }: { sitekey: string; onToken(token: string): void }) {
  const { t, locale } = useI18n();
  const element = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false), [attempt, setAttempt] = useState(0);
  const tokenCallback = useRef(onToken);
  tokenCallback.current = onToken;
  useEffect(() => {
    let cancelled = false, api: Turnstile | undefined, widget: string | undefined;
    setFailed(false);
    tokenCallback.current('');
    void load().then(service => {
      if (cancelled || !element.current) return;
      api = service;
      widget = api.render(element.current, {
        sitekey, action: 'login', theme: 'auto', size: element.current.clientWidth > 0 && element.current.clientWidth < 300 ? 'compact' : 'flexible', language: locale === 'zh' ? 'zh-CN' : 'en',
        callback: (token: string) => { if (!cancelled) { setFailed(false); tokenCallback.current(token); } },
        'expired-callback': () => { if (!cancelled) tokenCallback.current(''); },
        'error-callback': () => { if (!cancelled) { tokenCallback.current(''); setFailed(true); } },
        'timeout-callback': () => { if (!cancelled) { tokenCallback.current(''); setFailed(true); } },
      });
    }).catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; if (api && widget !== undefined) api.remove(widget); };
  }, [sitekey, locale, attempt]);
  return <div className="login-security" aria-label={t('安全校验', 'Security check')}>
    <div ref={element} />
    {failed && <p role="status">{t('安全校验未完成，请重试并允许 Cloudflare 校验加载。', 'Security check unavailable. Allow the Cloudflare check to load and try again.')} <button type="button" className="button" onClick={() => setAttempt(value => value + 1)}>{t('重试', 'Retry')}</button></p>}
  </div>;
}

export type LoginSecurity = { ready: boolean; payload(): { turnstile_token?: string }; reset(): void; widget: React.ReactNode };
export function useLoginSecurity(inherited?: LoginSecurity): LoginSecurity {
  const { capabilities } = useAccount(), { t } = useI18n();
  const sitekey = inherited ? undefined : capabilities?.turnstile_site_key;
  const [token, setToken] = useState(''), [version, setVersion] = useState(0);
  return inherited || {
    ready: !sitekey || Boolean(token),
    payload: () => {
      if (sitekey && !token) throw new Error(t('请先完成安全校验。', 'Complete the security check first.'));
      return sitekey ? { turnstile_token: token } : {};
    },
    reset: () => { if (sitekey) { setToken(''); setVersion(value => value + 1); } },
    widget: sitekey ? <LoginChallenge key={version} sitekey={sitekey} onToken={setToken} /> : null,
  };
}
