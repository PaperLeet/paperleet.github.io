import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { ShieldCheck } from "lucide-react";
import { useI18n } from "./i18n";
import { signInHref } from "./accountNavigation";
import { publicSite, hostedAppHref, hostedAppAvailable } from "./deployment";

export type Institution = { id: string; name: string; name_en?: string };
export type PublicUser = {
  id: string;
  username: string;
  display_name: string;
  avatar_color: string;
  bio: string;
  created_at: string;
  institution?: Institution | null;
  country?: string;
  show_progress?: boolean;
  is_test_user?: boolean;
};
export type AccountUser = PublicUser & {
  email: string;
  role: string;
  show_institution: boolean;
  show_country: boolean;
  show_progress: boolean;
  institution: Institution | null;
  institution_verified?: boolean;
  login_username?: string;
  country: string;
};
export type AccountCapabilities = {
  auth_available: boolean;
  auth_mode?: "institutional" | "test";
  test_login_available?: boolean;
  invite_only?: boolean;
  institution_verification_required: boolean;
  institutions: (Institution & { domains: string[] })[];
  reason?: string;
};
export class SocialError extends Error {
  status: number;
  code?: string;
  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
export async function socialApi<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    credentials: "same-origin",
    cache: "no-store",
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
  if (!response.ok)
    throw new SocialError(
      data.error || "Request failed",
      response.status,
      data.code,
    );
  return data as T;
}
export type DeploymentHealth = {
  mode: "local" | "hosted";
  judge: { ready: boolean; reason?: string };
  auth_required: boolean;
};
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
  const [health, setHealth] = useState<DeploymentHealth>();
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
  async function refresh() {
    if (publicSite) { setCurrentUser(null); setLoading(false); setError(""); return; }
    const epoch = ++accountEpoch.current;
    setLoading(true);
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
      setCurrentUser(null);
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
      const response = await fetch("/api/health", {
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
      setHealthError(errorMessage(e));
      setHealth((previous) =>
        previous
          ? {
              ...previous,
              judge: { ready: false, reason: "Service unavailable" },
            }
          : undefined,
      );
    }
  }
  useEffect(() => {
    void refresh();
    void refreshHealth();
    if (publicSite) return;
    const timer = setInterval(() => void refreshHealth(), 15000);
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
    };
  }, []);
  return (
    <AccountContext.Provider
      value={{
        user,
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
      {children}
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
  user: Pick<PublicUser, "display_name" | "avatar_color">;
  size?: number;
}) {
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
      {Array.from(user.display_name.trim())
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
export function InstitutionBadge({
  institution,
}: {
  institution?: Institution | null;
}) {
  const { t, locale } = useI18n();
  return institution ? (
    <span
      className="institution-badge"
      title={t("机构邮箱已验证", "Institutional email verified")}
    >
      <ShieldCheck size={13} />
      {locale === "en"
        ? institution.name_en || institution.name
        : institution.name}
    </span>
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
export function AccountNav({ returnTo = "" }: { returnTo?: string }) {
  const { user, loading } = useAccount();
  const { t } = useI18n();
  if (!hostedAppAvailable) return null;
  return user ? (
    <a
      className="account-nav"
      href={profileHref(user)}
      title={t("我的主页", "My profile")}
    >
      <Avatar user={user} size={29} />
    </a>
  ) : (
    <a className="account-nav sign-in-nav" href={hostedAppHref(signInHref(returnTo))}>
      {loading ? "…" : t("登录", "Sign in")}
    </a>
  );
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
