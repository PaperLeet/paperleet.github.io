import { createContext, lazy, Suspense, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useI18n } from "./i18n";
import { publicSite, hostedEdition } from "./deployment";
import { apiFetch } from "./apiClient";
import { initialDeploymentHealth, unavailableDeploymentHealth, type DeploymentHealth } from './deploymentHealth';
import { deferredFeature } from './deferredFeature';
export type { DeploymentHealth } from './deploymentHealth';

const HeaderAccountMenu = deferredFeature(() => import('./HeaderAccountMenu'), ['个人菜单','personal menu'], {loading:()=> <AccountMenuFallback/>,failure:(_props,retry)=><AccountMenuFallback retry={retry}/>});
import { apiUrl } from "./apiClient";
import { readProfileEditPolicy, type ProfileEditPolicy } from "./profileEditPolicyModel";
import { safeAvatarPath } from "./profileAvatarPath";
const AccountAccessGate = lazy(() => import('./Waitlist').then(module => ({ default: module.AccountAccessGate })));


export type Institution = { id: string; name: string; name_en?: string };
export type Affiliation = { name: string; name_en?: string; domain: string; kind: "institution" | "email_domain" };
export type PublicUser = {
  id: string;
  username: string;
  display_name: string;
  avatar_color: string;
  avatar_url?: string | null;
  bio: string;
  created_at: string;
  institution?: Institution | null;
  affiliation?: Affiliation | null;
  country?: string;
  show_progress?: boolean;
  is_test_user?: boolean;
  anonymous?: boolean;
};
export type AccountUser = PublicUser & {
  status?: 'active' | 'waitlisted' | 'disabled';
  email: string;
  role: string;
  show_institution: boolean;
  show_country: boolean;
  show_progress: boolean;
  institution: Institution | null;
  institution_verified?: boolean;
  affiliation_verified?: boolean;
  affiliation_email?: string | null;
  email_verified?: boolean;
  can_publish_community?: boolean;
  login_username?: string;
  country: string;
  profile_edit_policy?: ProfileEditPolicy;
};
export type AccountCapabilities = {
  admin_workspace?: boolean;
  admin_review_inbox?: boolean;
  admin_review_navigation?: boolean;
  community_taxonomy?: boolean;
  community_open_publishing?: boolean;
  community_reports?: boolean;
  community_report_history?: boolean;
  inbox_notifications?: boolean;
  inbox_read_all?: boolean;
  direct_messages?: boolean;
  problem_bookmarks?: boolean;
  study_lists?: boolean;
  community_anonymous_posts?: boolean;
  community_feed_sorting?: boolean;
  community_feed_period?: boolean;
  community_solutions?: boolean;
  community_profile_solutions?: boolean;
  community_profile_interviews?: boolean;
  community_papers?: boolean;
  account_achievements?: boolean;
  community_bookmarks?: boolean;
  interview_outcome_filter?: boolean;
  interview_context_filters?: boolean;
  auth_available: boolean;
  auth_mode?: "institutional" | "test" | "oauth";
  oauth_providers?: { id: "google" | "microsoft"; available: boolean }[];
  organization_account_required?: boolean;
  personal_account_waitlist?: boolean;
  test_login_available?: boolean;
  invite_only?: boolean;
  registration_policy?: "invite_only" | "work_email";
  work_email_registration?: boolean;
  email_verification_required?: boolean;
  email_login_available?: boolean;
  affiliation_verification_available?: boolean;
  email_waitlist_available?: boolean;
  email_login_policy?: "education_and_approved_domains" | "any_email";
  session_idle_days?: number;
  institution_verification_required: boolean;
  institutions: (Institution & { domains: string[] })[];
  reason?: string;
};
export class SocialError extends Error {
  status: number;
  code?: string;
  profileEditPolicy?: ProfileEditPolicy;
  profileEditField?: "username" | "display_name" | "avatar";
  nextChangeAt?: string;
  constructor(message: string, status: number, code?: string, details?: unknown) {
    super(message);
    this.status = status;
    this.code = code;
    if (details && typeof details === "object") {
      const data = details as Record<string, unknown>;
      this.profileEditPolicy = readProfileEditPolicy(data.profile_edit_policy);
      if (code === "profile_edit_limit" && ["username", "display_name", "avatar"].includes(String(data.field))) {
        this.profileEditField = data.field as "username" | "display_name" | "avatar";
        if (typeof data.next_change_at === "string" && Number.isFinite(Date.parse(data.next_change_at))) this.nextChangeAt = data.next_change_at;
      }
    }
  }
}
export async function socialApi<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await apiFetch(`/api/${path}`, {
    cache: "no-store",
    ...(signal ? { signal } : {}),
    ...(body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new SocialError(
      "The account and community service is unavailable. Please try again shortly.",
      response.status,
    );
  }
  if (!response.ok) {
    let message: string | undefined;
    try {
      // Localized publication/profile limits are only needed after their API rejection.
      if (data.code === "profile_edit_limit")
        message = (await import("./profileEditPolicy")).currentProfileEditLimitMessage(data);
      else if (data.code === "daily_quota_exceeded")
        message = (await import("./publicationQuota")).currentQuotaMessage(data);
    } catch { /* Preserve the API rejection if its optional wording cannot load. */ }
    throw new SocialError(message || data.error || "Request failed", response.status, data.code, data);
  }
  return data as T;
}
type AccountContextValue = {
  health?: DeploymentHealth;
  healthError: string;
  refreshHealth: () => Promise<void>;
  user: AccountUser | null;
  capabilities?: AccountCapabilities;
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  setUser: (user: AccountUser | null) => void;
};
const AccountContext = createContext<AccountContextValue>({
  healthError: "",
  refreshHealth: async () => {},
  user: null,
  loading: true,
  error: "",
  refresh: async () => {},
  setUser: () => {},
});
export function AccountProvider({ children }: { children: ReactNode }) {
  const [user, setCurrentUser] = useState<AccountUser | null>(null);
  const [capabilities, setCapabilities] = useState<AccountCapabilities>();
  const [health, setHealth] = useState<DeploymentHealth|undefined>(() => initialDeploymentHealth(hostedEdition));
  const [healthError, setHealthError] = useState("");
  const [loading, setLoading] = useState(true),
    [error, setError] = useState("");
  const accountEpoch = useRef(0);
  function setUser(next: AccountUser | null) {
    accountEpoch.current += 1;
    setCurrentUser(next);
    setLoading(false);
    setError("");
    try {
      localStorage.setItem(
        "paperleet.auth.changed",
        `${Date.now()}-${Math.random()}`,
      );
    } catch {
      /* The current tab still updates without browser storage. */
    }
  }
  async function refresh(background = false) {
    if (!hostedEdition) { setCurrentUser(null); setLoading(false); setError(""); return; }
    const epoch = ++accountEpoch.current;
    if (!background) setLoading(true);
    try {
      const data = await socialApi<{
        user: AccountUser | null;
        capabilities: AccountCapabilities;
      }>("account/me");
      if (epoch !== accountEpoch.current) return;
      setCurrentUser(data.user);
      setCapabilities(data.capabilities);
      setError("");
    } catch (e) {
      if (epoch !== accountEpoch.current) return;
      setCurrentUser(previous => previous?.status === 'waitlisted' ? previous : null);
      setError(errorMessage(e));
    } finally {
      if (epoch === accountEpoch.current) setLoading(false);
    }
  }
  async function refreshHealth() {
    if (publicSite) {
      setHealth({ mode: "local", auth_required: false, judge: { ready: false } });
      setHealthError("");
      return;
    }
    try {
      const response = await apiFetch("/api/health", {
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw Error("Service unavailable");
      const data = await response.json();
      if (data.mode !== "local" && data.mode !== "hosted")
        throw Error("Deployment configuration unavailable");
      setHealth({
        mode: data.mode,
        auth_required: data.auth_required === true,
        judge: data.judge || { ready: false },
      });
      setHealthError("");
    } catch (e) {
      setHealthError(hostedEdition ? errorMessage(e) : "");
      setHealth(previous => unavailableDeploymentHealth(previous,hostedEdition));
    }
  }
  useEffect(() => {
    void refresh();
    void refreshHealth();
    if (publicSite) return;
    const timer = setInterval(() => void refreshHealth(), 15000);
    let lastActivityRefresh = Date.now();
    const active = () => {
      if (document.visibilityState !== "visible" || Date.now() - lastActivityRefresh < 3600000) return;
      lastActivityRefresh = Date.now();
      void refresh(true);
    };
    window.addEventListener("pointerdown", active, { passive: true });
    window.addEventListener("keydown", active);
    window.addEventListener("focus", active);
    const changed = (event: StorageEvent) => {
      if (event.key === "paperleet.auth.changed") {
        setCurrentUser(null);
        void refresh();
      }
    };
    window.addEventListener("storage", changed);
    return () => {
      clearInterval(timer);
      window.removeEventListener("storage", changed);
      window.removeEventListener("pointerdown", active);
      window.removeEventListener("keydown", active);
      window.removeEventListener("focus", active);
    };
  }, []);
  return (
    <AccountContext.Provider
      value={{
        user: user?.status === 'waitlisted' ? null : user,
        capabilities,
        loading,
        error,
        refresh,
        setUser,
        health,
        healthError,
        refreshHealth,
      }}
    >
      {user?.status === 'waitlisted' ? <Suspense fallback={null}><AccountAccessGate user={user} refreshFailed={Boolean(error)} onRefresh={refresh} onSignOut={async () => {
        await socialApi('account/logout', {});
        setUser(null);
        location.hash = '#/account';
      }}>{children}</AccountAccessGate></Suspense> : children}
    </AccountContext.Provider>
  );
}
export const useAccount = () => useContext(AccountContext);
export const errorMessage = (e: unknown) =>
  e instanceof Error ? e.message : "Something went wrong. Please try again.";
export const profileHref = (user: Pick<PublicUser, "username">) =>
  `#/u/${encodeURIComponent(user.username)}`;
export function Avatar({
  user,
  size = 34,
}: {
  user: Pick<PublicUser, "display_name" | "avatar_color" | "avatar_url">;
  size?: number;
}) {
  const path = safeAvatarPath(user.avatar_url);
  const source = path ? apiUrl(path) : undefined;
  const [failed, setFailed] = useState<string>();
  const safeColor = /^#[0-9a-f]{6}$/i.test(user.avatar_color)
    ? user.avatar_color
    : "#4f7c73";
  return (
    <span
      className="social-avatar"
      style={{
        width: size,
        height: size,
        backgroundColor: safeColor,
        fontSize: size * 0.36,
      }}
      aria-label={user.display_name}
    >
      {source && failed !== source ? <img src={source} alt="" aria-hidden="true" loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(source)} /> : Array.from(user.display_name.trim())
        .slice(0, 1)
        .join("")
        .toLocaleUpperCase() || "P"}
    </span>
  );
}
export function PersonLink({
  user,
  compact = false,
}: {
  user: PublicUser;
  compact?: boolean;
}) {
  return (
    <a className="person-link" href={profileHref(user)}>
      <Avatar user={user} size={compact ? 24 : 34} />
      <span>{user.display_name}</span>
      <TestAccountBadge isTestUser={user.is_test_user} />
    </a>
  );
}
export function TestAccountBadge({ isTestUser }: { isTestUser?: boolean }) {
  const { t } = useI18n();
  return isTestUser ? (
    <span className="test-account-badge">{t("内测账号", "Test account")}</span>
  ) : null;
}
export function DateLabel({ value }: { value: string }) {
  const { locale } = useI18n();
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return (
    <time
      dateTime={value}
      title={date.toLocaleString(locale === "zh" ? "zh-CN" : "en-US")}
    >
      {date.toLocaleDateString(locale === "zh" ? "zh-CN" : "en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })}
    </time>
  );
}
export function AccountNav(props: import('./HeaderAccountMenu').HeaderAccountMenuProps) {
  return <HeaderAccountMenu {...props}/>;
}
function AccountMenuFallback({retry}:{retry?:()=>void}) {
  const { user } = useAccount();
  const { t } = useI18n();
  const label=retry?t('重试加载个人菜单','Retry personal menu'):t('正在加载个人菜单','Loading personal menu');
  return <button type="button" className="account-menu-trigger" disabled={!retry} onClick={retry} aria-label={label} title={label}>{user ? <Avatar user={user} size={32}/> : <span className="guest-avatar" aria-hidden="true">P</span>}</button>;
}
export function SocialNotice({
  error,
  children,
}: {
  error?: string;
  children?: ReactNode;
}) {
  return error ? (
    <div className="social-notice error" role="alert">
      {error}
    </div>
  ) : (
    <div className="social-notice" role="status">
      {children}
    </div>
  );
}
