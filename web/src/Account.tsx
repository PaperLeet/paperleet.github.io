import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  LogOut,
  Mail,
  ShieldCheck,
  KeyRound,
  UserRound,
} from "lucide-react";
import { useI18n } from "./i18n";
import {
  Avatar,
  AffiliationBadge,
  SocialNotice,
  SocialError,
  TestAccountBadge,
  errorMessage,
  profileHref,
  socialApi,
  useAccount,
} from "./social";
import type { AccountUser } from "./social";
import type { AccountCapabilities } from "./social";
import { organizationLoginError } from "./accountNavigation";
import { decodeProblemHash } from "./reports";
import { AvatarImageError, prepareProfileAvatar } from "./profileAvatar";
import { readProfileEditPolicy } from "./profileEditPolicy";

const COLORS = [
  "#4f7c73",
  "#8665a1",
  "#5b78a6",
  "#ac7545",
  "#a45e78",
  "#627a43",
  "#747789",
  "#9b6853",
];
export default function Account({ returnTo }: { returnTo?: string }) {
  const { user, loading, capabilities, error, refresh } = useAccount();
  const { t } = useI18n();
  useEffect(() => {
    if (user && !loading && returnTo) location.hash = returnTo;
  }, [user?.id, loading, returnTo]);
  return (
    <main className="social-page account-page">
      <a
        className="social-back"
        href={returnTo || (user ? profileHref(user) : "#/community")}
      >
        <ArrowLeft size={14} />
        {returnTo ? t("返回刚才的页面", "Back to where you left off") : user
          ? t("返回个人主页", "Back to profile")
          : t("返回社区", "Back to community")}
      </a>
      {returnTo && !user && <p className="social-description">{t("登录成功后将返回刚才的页面。", "After signing in, you will return to where you left off.")}</p>}
      {loading ? (
        <div className="social-loading">
          <span className="spinner" />
          {t("正在加载账户…", "Loading account…")}
        </div>
      ) : error ? (
        <>
          <SocialNotice error={error} />
          <button className="button" onClick={() => void refresh()}>
            {t("重试", "Retry")}
          </button>
        </>
      ) : user ? (
        <ProfileSettings key={user.id} user={user} />
      ) : capabilities?.auth_mode === "test" ? (
        <TestSignIn enabled={capabilities.auth_available === true} />
      ) : capabilities?.auth_mode === "oauth" ? (
        <OrganizationSignIn providers={capabilities.oauth_providers || []} returnTo={returnTo}
          emailLoginAvailable={capabilities.email_login_available}
          loginError={typeof location === "undefined" ? undefined : organizationLoginError(decodeProblemHash(location.hash))} />
      ) : (
        <SignIn enabled={capabilities?.auth_available === true} workEmailRegistration={capabilities?.work_email_registration === true} />
      )}
    </main>
  );
}

export function OrganizationSignIn({ providers, returnTo, loginError, emailLoginAvailable }: {
  providers: NonNullable<AccountCapabilities["oauth_providers"]>; returnTo?: string; loginError?: string; emailLoginAvailable?: boolean;
}) {
  const { t } = useI18n();
  const [busy, setBusy] = useState<string>(), [error, setError] = useState("");
  const failures: Record<string, string> = {
    identity: t("未能完成身份验证，请重新选择账号。若此邮箱曾使用其他登录方式注册，请使用原来的登录方式。", "We could not verify this sign-in. Choose an account again. If this email was registered with another sign-in method, use the original method."),
    expired: t("此次登录已过期，请重新开始。", "This sign-in expired. Please start again."),
    cancelled: t("你已取消登录，可以重新选择账号。", "Sign-in was cancelled. You can choose an account again."),
    unavailable: t("登录服务暂时不可用，请稍后重试。", "The sign-in service is temporarily unavailable. Please try again shortly."),
    rate_limited: t("登录尝试过于频繁，请稍后重试。", "Too many sign-in attempts. Please try again shortly."),
  };
  async function signIn(provider: "google" | "microsoft") {
    if (busy) return;
    setBusy(provider); setError("");
    try {
      const result = await socialApi<{ url: string }>(`account/oauth/${provider}/start`, { return_to: returnTo || "#/" });
      const target = new URL(result.url);
      const expected = provider === "google" ? "https://accounts.google.com" : "https://login.microsoftonline.com";
      if (target.origin !== expected) throw new Error("The sign-in service returned an invalid destination.");
      location.assign(target.href);
    } catch (e) { setError(errorMessage(e)); setBusy(undefined); }
  }
  return <><div className="account-grid">
    <section className="social-panel login-card">
      <span className="social-eyebrow">PAPERLEET</span>
      <h1>{t("登录 PaperLeet", "Sign in to PaperLeet")}</h1>
      <p className="social-description">{t("使用 Google 或 Microsoft 账号登录，首次登录将创建 PaperLeet 账号，无需接收本站验证码。", "Sign in with Google or Microsoft. Your first sign-in creates a PaperLeet account without an emailed code from us.")}</p>
      {(error || (loginError && failures[loginError])) && <SocialNotice error={error || failures[loginError!]} />}
      <div className="organization-login-buttons" aria-busy={Boolean(busy)}>
        {(["google", "microsoft"] as const).map(provider => {
          const available = providers.some(item => item.id === provider && item.available);
          return <button key={provider} type="button" className="organization-login-button" disabled={!available || Boolean(busy)} onClick={() => void signIn(provider)}>
            {provider === "google" ? <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path fill="#4285F4" d="M21.6 12.23c0-.71-.06-1.39-.18-2.05H12v3.88h5.38a4.6 4.6 0 0 1-2 3.02v2.51h3.24c1.89-1.74 2.98-4.3 2.98-7.36Z"/><path fill="#34A853" d="M12 22c2.7 0 4.96-.9 6.62-2.41l-3.24-2.51c-.9.6-2.05.96-3.38.96-2.6 0-4.81-1.76-5.6-4.12H3.05v2.59A10 10 0 0 0 12 22Z"/><path fill="#FBBC05" d="M6.4 13.92a6 6 0 0 1 0-3.84V7.49H3.05a10 10 0 0 0 0 9.02l3.35-2.59Z"/><path fill="#EA4335" d="M12 5.96c1.47 0 2.79.51 3.83 1.51l2.87-2.87A9.62 9.62 0 0 0 12 2a10 10 0 0 0-8.95 5.49l3.35 2.59A5.99 5.99 0 0 1 12 5.96Z"/></svg> : <svg viewBox="0 0 21 21" width="20" height="20" aria-hidden="true"><path fill="#f25022" d="M0 0h10v10H0z"/><path fill="#7fba00" d="M11 0h10v10H11z"/><path fill="#00a4ef" d="M0 11h10v10H0z"/><path fill="#ffb900" d="M11 11h10v10H11z"/></svg>}
            <span>{busy === provider ? t("正在前往登录…", "Opening sign-in…") : provider === "google" ? t("使用 Google 登录", "Sign in with Google") : t("使用 Microsoft 登录", "Sign in with Microsoft")}</span>
            {!available && <small>{t("尚未启用", "Not enabled yet")}</small>}
          </button>;
        })}
      </div>
      <p className="social-description">{t("目前仅向学校／企业账户和已获邀请的账户开放使用。个人账户也可以注册，将先加入等待名单，后续开放后即可使用。", "Access is currently open to school/company accounts and invited accounts. Personal accounts can also register and join the waitlist for future access.")}</p>
    </section>
    <aside className="social-panel login-card">
      <h2>{t("哪些邮箱可以使用？", "Which accounts work?")}</h2>
      <p className="social-description">{t("Google Workspace 和 Microsoft 365／Entra 管理的学校、企业账号可以直接使用。个人 Gmail、Outlook 等账号登录后会进入等待名单；已获邀请的账户按现有资格使用。部分机构可能要求管理员批准应用。", "Managed Google Workspace and Microsoft 365/Entra school/company accounts receive access. Personal Gmail, Outlook and similar accounts join the waitlist; invited accounts retain their existing access. Some organizations require administrator approval.")}</p>
      <p className="social-description">{t("登录通过 Google 或 Microsoft 完成。可用账号的注册机构或邮箱域名会公开展示，完整邮箱保持私密；等待名单账号不会公开个人主页。组织账号不代表学校或雇主对个人身份的背书。", "Sign-in is handled by Google or Microsoft. Active accounts show their registration institution or email domain publicly; full email addresses stay private. Waitlisted accounts have no public profile. An organization account is not an endorsement by a school or employer.")}</p>
      {emailLoginAvailable === true && <p className="social-description">{t("学校使用其他邮件系统时，可以通过下方邮箱验证码入口登录。公司邮箱需先审核域名。完整邮箱不会出现在公开主页。", "Schools using other email systems can use the email-code option below. Company domains require review first. Your full email stays private.")}</p>}
      <p className="account-fineprint">{t("登录后会自动保持登录；连续 90 天未使用、主动退出或账号权限被撤销时，需要重新登录。请勿在公共电脑上保留登录。", "Sign-in is renewed while you use PaperLeet. After 90 days of inactivity, signing out, or access revocation, you must sign in again. Sign out on shared computers.")}</p>
    </aside>
  </div>
    {emailLoginAvailable === true && <details className="email-login-alternative">
      <summary>{t("使用学校／已审核公司邮箱验证码登录", "Use a school / reviewed company email code")}</summary>
      <SignIn enabled={emailLoginAvailable === true} workEmailRegistration onChooseProvider={provider => void signIn(provider)} providers={providers} />
    </details>}
  </>;
}
export function TestSignIn({ enabled }: { enabled: boolean }) {
  const { t } = useI18n();
  const { setUser, health } = useAccount();
  const [username, setUsername] = useState(""),
    [accessCode, setAccessCode] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function signIn() {
    if (!enabled) return;
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<{ user: AccountUser }>(
        "account/test-login",
        { username: username.trim(), access_code: accessCode },
      );
      setAccessCode("");
      setUser(data.user);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="account-grid">
      <section className="social-panel login-card">
        <span className="social-eyebrow">PAPERLEET BETA</span>
        <h1>{t("内测账号登录", "Test account sign-in")}</h1>
        <p className="social-description">
          {t(
            "使用管理员提供的账号和访问码，开始答题并参与社区。",
            "Use the account and access code provided by an administrator to practice and join the community.",
          )}
        </p>
        {!enabled && (
          <SocialNotice>
            {t(
              "此部署的内测登录尚未就绪，请联系管理员。",
              "Test account sign-in is not ready on this deployment. Please contact an administrator.",
            )}
          </SocialNotice>
        )}
        {error && <SocialNotice error={error} />}
        <form
          className="social-form"
          onSubmit={(event) => {
            event.preventDefault();
            void signIn();
          }}
        >
          <label>
            {t("内测账号", "Test username")}
            <input
              autoComplete="username"
              required
              maxLength={80}
              value={username}
              disabled={busy || !enabled}
              placeholder={t(
                "管理员提供的登录账号",
                "Username from your administrator",
              )}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label>
            {t("访问码", "Access code")}
            <input
              type="password"
              autoComplete="current-password"
              required
              maxLength={256}
              value={accessCode}
              disabled={busy || !enabled}
              placeholder={t("输入你的访问码", "Enter your access code")}
              onChange={(event) => setAccessCode(event.target.value)}
            />
          </label>
          <button className="button" type="submit" disabled={busy || !enabled}>
            {busy ? <span className="spinner" /> : <KeyRound size={15} />}
            {t("登录内测", "Sign in to the beta")}
          </button>
        </form>
        <p className="account-fineprint">
          {health?.mode === "hosted"
            ? t(
                "内测账号由管理员创建。昵称和头像始终公开；每个账号的代码草稿和判题结果独立保存。",
                "Test accounts are created by an administrator. Display names and avatars are always public; code drafts and judging results are kept separate for each account.",
              )
            : t(
                "内测账号由管理员创建，昵称和头像始终公开。本地模式的练习记录保存在当前浏览器中。",
                "Test accounts are created by an administrator. Display names and avatars are always public. Guest practice is not saved after the page closes or reloads.",
              )}
        </p>
      </section>
      <aside className="account-aside">
        <section className="social-panel">
          <UserRound size={27} className="accent-icon" />
          <h3>{t("还没有内测账号？", "Need a test account?")}</h3>
          <p>
            {t(
              "请向管理员申请账号和访问码，收到后即可登录，无需填写邮箱。",
              "Ask an administrator for a username and access code. You can sign in as soon as you receive them, without providing an email address.",
            )}
          </p>
        </section>
        <section className="social-panel">
          <TestAccountBadge isTestUser />
          <h3>{t("内测身份", "Beta identity")}</h3>
          <p>
            {t(
              "主页会标注内测账号，不显示机构邮箱认证。国家／地区仍由你选择是否公开。",
              "Your profile is labeled as a test account, with no institutional email verification claim. You still choose whether to display your country or region.",
            )}
          </p>
        </section>
      </aside>
    </div>
  );
}
export function EmailQuotaNotice({ email, waitlistAvailable, onChooseProvider, providers = [] }: {
  email: string; waitlistAvailable: boolean;
  onChooseProvider?: (provider: "google" | "microsoft") => void;
  providers?: NonNullable<AccountCapabilities["oauth_providers"]>;
}) {
  const { t } = useI18n();
  const [joining, setJoining] = useState(false), [joined, setJoined] = useState(false), [error, setError] = useState("");
  async function join() {
    if (joining || joined) return;
    setJoining(true); setError("");
    try {
      const result = await socialApi<{ joined: boolean }>("account/waitlist", { email: email.trim() });
      if (!result.joined) throw new Error(t("候补登记未完成，请重试。", "Waitlist signup did not complete. Please retry."));
      setJoined(true);
    } catch (e) { setError(errorMessage(e)); }
    finally { setJoining(false); }
  }
  return <div className="email-quota-notice">
    <SocialNotice>{joined
      ? t("已加入 waitlist，邮箱已登记。你仍可使用 Google／Microsoft 登录。", "You're on the waitlist. Your email has been recorded. You can still sign in with Google/Microsoft.")
      : t("当前邮箱验证码名额已满。可加入 waitlist，或使用 Google／Microsoft 登录。", "Email sign-in capacity is full. Join the waitlist, or sign in with Google/Microsoft.")}</SocialNotice>
    <div className="email-login-dialog-actions">
      {waitlistAvailable && <button className="button subtle" type="button" disabled={joining || joined} onClick={() => void join()}>
        {joined ? t("已加入 waitlist", "On the waitlist") : joining ? t("正在登记…", "Joining…") : t("加入 waitlist", "Join waitlist")}
      </button>}
      {onChooseProvider && providers.filter(provider => provider.available).map(provider => <button className="button" type="button" key={provider.id} onClick={() => onChooseProvider(provider.id)}>
        {provider.id === "google" ? t("使用 Google 登录", "Sign in with Google") : t("使用 Microsoft 登录", "Sign in with Microsoft")}
      </button>)}
    </div>
    {error && <SocialNotice error={error} />}
  </div>;
}

export function SignIn({ enabled, workEmailRegistration = false, onChooseProvider, providers = [] }: {
  enabled: boolean; workEmailRegistration?: boolean;
  onChooseProvider?: (provider: "google" | "microsoft") => void;
  providers?: NonNullable<AccountCapabilities["oauth_providers"]>;
}) {
  const { t, locale } = useI18n();
  const { setUser, capabilities } = useAccount();
  const [email, setEmail] = useState(""),
    [code, setCode] = useState(""),
    [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const [challenge, setChallenge] = useState("");
  const [remaining, setRemaining] = useState(0);
  const [recommend, setRecommend] = useState(false);
  const [preferEmail, setPreferEmail] = useState(false);
  const [quotaReached, setQuotaReached] = useState(false);
  const recommendation = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (recommend) recommendation.current?.showModal();
    else recommendation.current?.close();
  }, [recommend]);
  useEffect(() => {
    if (!remaining) return;
    const timer = setTimeout(() => setRemaining(value => Math.max(0, value - 1)), 1000);
    return () => clearTimeout(timer);
  }, [remaining]);
  const domain = email.trim().toLowerCase().split("@")[1];
  const institution = capabilities?.institutions?.find((i) =>
    i.domains.includes(domain),
  );
  const eligible = Boolean(domain && (domain.endsWith(".edu.cn") || domain.endsWith(".edu") || institution));
  async function sendCode(confirmed = preferEmail) {
    if (!enabled || busy || remaining > 0) return;
    if (!eligible) { setError(t("早期仅支持 edu.cn、.edu 学校邮箱和已审核的学校／公司域名。请使用 Google／Microsoft，或在下方申请域名审核。", "Email codes currently support edu.cn, .edu and reviewed school/company domains. Use Google/Microsoft or request domain review below.")); return; }
    if (domain?.endsWith(".edu") && !confirmed) { setRecommend(true); return; }
    setBusy(true);
    setQuotaReached(false);
    setError("");
    setMessage("");
    try {
      const result = await socialApi<{ sent: boolean; oauth_recommended?: boolean; challenge?: string; retry_after?: number }>("account/send-code", { email: email.trim(), prefer_email: confirmed });
      if (result.oauth_recommended) { setRecommend(true); return; }
      if (!result.sent) throw new Error(t("验证码未发送，请稍后重试。", "The code was not sent. Please try again."));
      setChallenge(result.challenge || "");
      setCode("");
      setRemaining(result.retry_after || 60);
      setSent(true);
      setMessage(
        t(
          "验证码已发送，请检查学校或公司邮箱。",
          "A verification code has been sent to your school or company email.",
        ),
      );
    } catch (e) {
      const exhausted = e instanceof SocialError && e.code === "email_quota_exhausted";
      setQuotaReached(exhausted);
      setError(exhausted ? "" : e instanceof SocialError && e.status === 429
        ? t("验证码请求过于频繁，请稍后重试，或使用上方 Google／Microsoft 登录。", "Too many code requests. Try again later, or use Google/Microsoft above.")
        : e instanceof SocialError && e.status === 503
          ? t("邮箱服务暂时不可用，请稍后重试，或使用 Google／Microsoft 登录。", "Email sign-in is temporarily unavailable. Try again later, or use Google/Microsoft.")
          : errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function verify() {
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<{ user: AccountUser }>("account/verify", {
        email: email.trim(),
        code: code.trim(),
        challenge,
      });
      setUser(data.user);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="account-grid">
      <section className="social-panel login-card">
        <span className="social-eyebrow">PAPERLEET ACCOUNT</span>
        <h1>{t("加入研究者社区", "Join the community")}</h1>
        <p className="social-description">
          {t(
            "使用学校邮箱或已审核的公司邮箱接收验证码，注册／登录 PaperLeet。",
            "Register or sign in to PaperLeet with a school or reviewed company email code.",
          )}
        </p>
        {capabilities?.invite_only && (
          <SocialNotice>
            {t(
              "当前为邀请制内测。机构邮箱验证和管理员邀请均通过后才能登录；需要加入时请联系维护者。",
              "This beta is invitation-only. Sign-in requires both an administrator invitation and a verified institutional email. Contact the maintainers to request access.",
            )}
          </SocialNotice>
        )}
        {workEmailRegistration && (
          <SocialNotice>
            {t("无需邀请码：edu.cn 学校邮箱可直接发送；.edu 优先使用 Google／Microsoft，也可继续收取验证码。其他学校和公司邮箱需审核域名。Gmail、QQ、Outlook 等个人邮箱不支持验证码。", "No invitation needed: send directly to edu.cn school addresses. For .edu, try Google/Microsoft first, or continue with a code. Other school and company domains require review. Personal services such as Gmail, QQ and Outlook cannot use email codes.")}
          </SocialNotice>
        )}
        {!enabled && (
          <SocialNotice>
            {t(
              "此部署尚未启用邮箱登录。你可以浏览社区；发帖和关注需在管理员配置邮件验证后使用。",
              "Email sign-in is not enabled on this deployment yet. You can browse the community; posting and following will be available after the administrator configures verification.",
            )}
          </SocialNotice>
        )}
        {error && <SocialNotice error={error} />}
        {quotaReached && <EmailQuotaNotice key={email.trim().toLowerCase()} email={email}
          waitlistAvailable={capabilities?.email_waitlist_available === true}
          onChooseProvider={onChooseProvider} providers={providers} />}
        {message && <SocialNotice>{message}</SocialNotice>}
        <form
          className="social-form"
          onSubmit={(e) => {
            e.preventDefault();
            void (sent ? verify() : sendCode());
          }}
        >
          <label>
            {t("学校／公司邮箱", "School / company email")}
            <input
              type="email"
              autoComplete="email"
              required
              maxLength={254}
              value={email}
              disabled={busy || sent || !enabled}
              placeholder="you@university.edu.cn"
              onChange={(e) => { setEmail(e.target.value); setPreferEmail(false); setChallenge(""); }}
            />
          </label>
          {institution && (
            <div className="recognized-institution">
              <ShieldCheck size={14} />
              {locale === "en"
                ? institution.name_en || institution.name
                : institution.name}
            </div>
          )}
          {sent && (
            <label>
              {t("邮件验证码", "Email verification code")}
              <input
                autoComplete="one-time-code"
                inputMode="numeric"
                required
                value={code}
                maxLength={12}
                autoFocus
                disabled={busy}
                onChange={(e) => setCode(e.target.value)}
                placeholder={t(
                  "输入邮件中的验证码",
                  "Enter the code from your email",
                )}
              />
            </label>
          )}
          <button className="button" type="submit" disabled={busy || !enabled || (!sent && remaining > 0)}>
            {busy ? (
              <span className="spinner" />
            ) : sent ? (
              <ShieldCheck size={15} />
            ) : (
              <Mail size={15} />
            )}
            {sent
              ? t("验证并登录", "Verify and sign in")
              : remaining > 0 ? t(`${remaining} 秒后可重发`, `Resend in ${remaining}s`) : t("发送验证码", "Send verification code")}
          </button>
          {sent && (
            <div className="form-inline-actions">
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setSent(false);
                  setCode("");
                  setChallenge("");
                  setPreferEmail(false);
                  setMessage("");
                }}
              >
                {t("更换邮箱", "Change email")}
              </button>
              <button
                type="button"
                disabled={busy || remaining > 0}
                onClick={() => void sendCode()}
              >
                {remaining > 0 ? t(`${remaining} 秒后可重发`, `Resend in ${remaining}s`) : t("重新发送", "Resend code")}
              </button>
            </div>
          )}
        </form>
        <dialog ref={recommendation} className="email-login-dialog" aria-labelledby="email-login-recommendation" onCancel={() => setRecommend(false)}>
          <h2 id="email-login-recommendation">{t("建议使用学校的 Google／Microsoft 登录", "Try your school's Google / Microsoft sign-in")}</h2>
          <p>{t(".edu 学校通常提供组织账号登录，无需等待邮件。若学校使用其他邮件系统，你也可以继续发送验证码。", ".edu schools often support organization sign-in without waiting for email. If your school uses another mail system, you can continue with an email code.")}</p>
          <div className="email-login-dialog-actions">
            {onChooseProvider && providers.filter(provider => provider.available).map(provider => <button className="button" type="button" key={provider.id} onClick={() => { setRecommend(false); onChooseProvider(provider.id); }}>{provider.id === "google" ? t("使用 Google 登录", "Sign in with Google") : t("使用 Microsoft 登录", "Sign in with Microsoft")}</button>)}
            <button className="button subtle" type="button" onClick={() => { setRecommend(false); setPreferEmail(true); void sendCode(true); }}>{t("仍然发送邮箱验证码", "Continue with email code")}</button>
            <button className="text-button" type="button" onClick={() => setRecommend(false)}>{t("返回", "Back")}</button>
          </div>
        </dialog>
        <p className="account-fineprint">
          {workEmailRegistration ? t(
            "验证码 10 分钟有效，只能使用一次。已有 Google／Microsoft 账户请继续使用原方式登录，暂不自动合并不同登录方式。昵称、头像和注册机构或邮箱域名会公开，完整邮箱保持私密。",
            "Codes expire in 10 minutes and work once. If you already use Google/Microsoft, continue with that method; sign-in identities are not automatically merged. Your display name, avatar and registration institution or email domain are public; your full email stays private.",
          ) : t(
            "只有通过审核的机构邮箱可以注册。昵称、头像和注册邮箱归属始终公开，完整邮箱保持私密；国家／地区由你选择是否展示。",
            "Registration is limited to approved institutional email domains. Your display name, avatar, and registration affiliation are public; your full email stays private. You choose whether to display your country or region.",
          )}
        </p>
      </section>
      <aside className="account-aside">
        <div className="social-panel">
          <ShieldCheck size={27} className="accent-icon" />
          <h3>{t("学校／公司邮箱验证", "School / company email verification")}</h3>
          <p>
            {t(
              "验证说明你拥有这个机构邮箱，不代表在读或在职身份认证。机构名称或邮箱域名会公开，完整邮箱不会出现在公开主页。",
              "Verification confirms access to an institutional mailbox. It does not certify current student or employment status. Your institution or email domain is public; your full email address is never shown on your public profile.",
            )}
          </p>
        </div>
        <InstitutionRequest openRegistration={workEmailRegistration} />
      </aside>
    </div>
  );
}
function InstitutionRequest({ openRegistration }: { openRegistration: boolean }) {
  const { t } = useI18n();
  const [name, setName] = useState(""),
    [website, setWebsite] = useState(""),
    [domain, setDomain] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [done, setDone] = useState(false);
  async function submit() {
    setBusy(true);
    setError("");
    try {
      await socialApi("account/institution-requests", {
        name: name.trim(),
        website: website.trim(),
        email_domain: domain.trim(),
      });
      setDone(true);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="social-panel institution-request">
      <summary>
        {t("找不到你的机构？", "Institution not listed?")}
        <ChevronDown size={15} />
      </summary>
      <p>
        {openRegistration ? t(
          "edu.cn 和 .edu 学校邮箱可按提示登录。其他学校和公司邮箱请先提交域名审核；审核通过后才能发送验证码。",
          "Follow the sign-in prompts for edu.cn and .edu schools. Other school and company domains must be reviewed before they can receive codes.",
        ) : t(
          "提供机构信息，管理员审核域名后即可使用邮箱验证。",
          "Suggest your institution. Email verification becomes available once an administrator approves its domain.",
        )}
      </p>
      {error && <SocialNotice error={error} />}
      {done ? (
        <SocialNotice>
          {t(
            "申请已提交，正在等待管理员审核。",
            "Your request has been submitted for administrator review.",
          )}
        </SocialNotice>
      ) : (
        <form
          className="social-form"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label>
            {t("机构名称", "Institution name")}
            <input
              required
              maxLength={160}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            {t("官方网站", "Official website")}
            <input
              type="url"
              required
              maxLength={500}
              placeholder="https://university.edu"
              value={website}
              onChange={(e) => setWebsite(e.target.value)}
            />
          </label>
          <label>
            {t("邮箱域名", "Email domain")}
            <input
              required
              maxLength={253}
              placeholder="university.edu"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            />
          </label>
          <button className="button subtle" disabled={busy}>
            {t("申请收录", "Request review")}
            <ArrowRight size={14} />
          </button>
        </form>
      )}
    </details>
  );
}
export function ProfileSettings({ user }: { user: AccountUser }) {
  const { t, locale } = useI18n();
  const { setUser, health } = useAccount();
  const [name, setName] = useState(user.display_name),
    [username, setUsername] = useState(user.username),
    [color, setColor] = useState(user.avatar_color),
    [bio, setBio] = useState(user.bio || "");
  const [country, setCountry] = useState(user.country || ""),
    [showCountry, setShowCountry] = useState(user.show_country),
    [showProgress, setShowProgress] = useState(user.show_progress);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [policy, setPolicy] = useState(() => readProfileEditPolicy(user.profile_edit_policy));
  const [avatarImage, setAvatarImage] = useState<string | null>();
  const [preparingAvatar, setPreparingAvatar] = useState(false);
  const active = useRef(true);
  const imageEpoch = useRef(0);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; imageEpoch.current += 1; };
  }, []);
  const avatarSupported = !!policy;
  const changing = busy || preparingAvatar;
  const dateLabel = (date: string) => new Date(date).toLocaleString(locale === "en" ? "en-US" : "zh-CN", { timeZoneName: "short" });
  function changeHint(field: "display_name" | "username") {
    const days = field === "display_name" ? 7 : 30;
    if (!policy) return t(`每 ${days} 天修改一次的限制将在服务更新后启用。`, `The once-every-${days}-days limit will be enabled after a service update.`);
    const next = policy[field].next_change_at;
    return next && Date.parse(next) > Date.now()
      ? t(`每 ${days} 天可修改一次。下次可修改：${dateLabel(next)}`, `Change once every ${days} days. Next change: ${dateLabel(next)}`)
      : t(`每 ${days} 天可修改一次，目前可以修改。`, `Change once every ${days} days. A change is available now.`);
  }
  async function selectAvatar(file: File) {
    if (!avatarSupported || busy) return;
    const epoch = ++imageEpoch.current;
    setPreparingAvatar(true);
    setError("");
    try {
      const image = await prepareProfileAvatar(file);
      if (active.current && epoch === imageEpoch.current) setAvatarImage(image);
    } catch (e) {
      if (!active.current || epoch !== imageEpoch.current) return;
      const code = e instanceof AvatarImageError ? e.code : "decode";
      setError(code === "type" ? t("请选择 JPEG、PNG 或 WebP 图片。", "Choose a JPEG, PNG, or WebP image.")
        : code === "size" ? t("头像原图须大于 0 字节且不超过 5 MiB。", "The original avatar image must be nonempty and no larger than 5 MiB.")
        : t("无法读取这张图片，请换一张 JPEG、PNG 或 WebP 图片。", "This image could not be read. Try another JPEG, PNG, or WebP file."));
    } finally {
      if (active.current && epoch === imageEpoch.current) setPreparingAvatar(false);
    }
  }
  async function save() {
    if (changing) return;
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<{ user: AccountUser }>("account/profile", {
        display_name: name.trim(),
        username: username.trim(),
        avatar_color: color,
        ...(avatarSupported && avatarImage !== undefined ? { avatar_image: avatarImage } : {}),
        bio,
        country: country.trim().toUpperCase(),
        show_institution: true,
        show_country: showCountry,
        show_progress: showProgress,
        expected_user_id: user.id,
      });
      if (!active.current) return;
      if (data.user.id !== user.id) throw new Error(t("账号已发生变化，请重新打开设置后再保存。", "Your account changed. Reopen settings before saving."));
      setUser(data.user);
      location.hash = profileHref(data.user);
    } catch (e) {
      if (!active.current) return;
      if (e instanceof SocialError && e.profileEditPolicy) setPolicy(e.profileEditPolicy);
      setError(errorMessage(e));
    } finally {
      if (active.current) setBusy(false);
    }
  }
  async function logout(allDevices = false) {
    setBusy(true);
    setError("");
    try {
      await socialApi(allDevices ? "account/logout-all" : "account/logout", {});
      setUser(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="social-page-heading">
        <div>
          <h1>{t("个人资料与隐私", "Profile & privacy")}</h1>
          <div className="profile-workspace-links"><a className="text-button profile-workspace-settings" href="#/settings">{t('工作区设置与代码备份', 'Workspace settings & code backup')}<ArrowRight size={14} aria-hidden="true" /></a>
          <a className="text-button profile-workspace-settings" href="#/achievements">{t('我的学习勋章', 'My learning achievements')}<ArrowRight size={14} aria-hidden="true" /></a></div>
        </div>
        <a className="button subtle" href={profileHref(user)}>
          {t("查看公开主页", "View public profile")}
          <ArrowRight size={14} />
        </a>
      </div>
      <div className="account-grid">
        <form
          className="social-panel social-form profile-settings"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          {error && <SocialNotice error={error} />}
          <div className="settings-avatar">
            <ProfileAvatarPreview user={{ ...user, display_name: name, avatar_color: color }} image={avatarImage} size={64} />
            <div>
              <h3>{t("公开头像", "Public avatar")}</h3>
              <div className="avatar-upload-actions">
                <label className={`button subtle avatar-file-button ${!avatarSupported || changing ? "disabled" : ""}`}>
                  {preparingAvatar ? t("正在处理图片…", "Preparing image…") : t("选择图片", "Choose image")}
                  <input type="file" accept="image/jpeg,image/png,image/webp" aria-label={t("选择头像图片", "Choose avatar image")}
                    disabled={!avatarSupported || changing} onChange={(event) => {
                      const file = event.currentTarget.files?.[0];
                      event.currentTarget.value = "";
                      if (file) void selectAvatar(file);
                    }} />
                </label>
                {(typeof avatarImage === "string" || (avatarImage === undefined && user.avatar_url)) && <button type="button" className="button subtle" disabled={!avatarSupported || changing}
                  onClick={() => { imageEpoch.current += 1; setAvatarImage(null); }}>{t("移除图片", "Remove image")}</button>}
                {(avatarImage !== undefined || color !== user.avatar_color) && <button type="button" className="button subtle" disabled={changing}
                  onClick={() => { imageEpoch.current += 1; setAvatarImage(undefined); setColor(user.avatar_color); }}>{t("撤销头像更改", "Discard avatar change")}</button>}
              </div>
              <p className="avatar-upload-help">{avatarSupported
                ? t("JPEG / PNG / WebP，原图不超过 5 MiB。居中裁为方形；点击保存设置后才上传。", "JPEG / PNG / WebP, up to 5 MiB. Center-cropped to a square; uploaded only when you save changes.")
                : t("头像上传需等待服务更新；其他现有资料仍可保存。", "Avatar uploads are waiting for a service update. You can still save existing profile fields.")}</p>
              {policy && <p className="profile-edit-hint">{t(`头像每天最多修改 ${policy.avatar.daily_limit} 次，今天剩余 ${policy.avatar.remaining} 次，UTC 00:00 重置。图片和颜色在同次保存中合计一次。`, `Avatar changes: up to ${policy.avatar.daily_limit} per day, ${policy.avatar.remaining} remaining today. Resets at 00:00 UTC. An image and color saved together count as one change.`)}</p>}
              <div className="avatar-palette">
                {COLORS.map((c) => (
                  <button
                    key={c}
                    type="button"
                    aria-label={`${t("头像颜色", "Avatar color")} ${c}`}
                    aria-pressed={color === c}
                    style={{ backgroundColor: c }}
                    disabled={changing}
                    onClick={() => setColor(c)}
                  >
                    {color === c && <Check size={14} />}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <label>
            {t("公开昵称", "Public display name")}
            <input
              required
              minLength={1}
              maxLength={40}
              value={name}
              disabled={changing}
              onChange={(e) => setName(e.target.value)}
            />
            <small className="profile-edit-hint">{changeHint("display_name")}</small>
          </label>
          <label>
            {t("用户名", "Username")}
            <input
              required
              minLength={3}
              maxLength={24}
              pattern="[a-z0-9_]{3,24}"
              value={username}
              disabled={changing}
              onChange={(e) => setUsername(e.target.value.toLowerCase())}
            />
            <small>
              {t(
                "3–24 位小写字母、数字或下划线，用于个人主页链接。",
                "3–24 lowercase letters, numbers, or underscores. Used in your profile link.",
              )}
            </small>
            <small className="profile-edit-hint">{changeHint("username")}</small>
            <small>{t("修改昵称或用户名不会改变内部账号 ID、已保存代码和提交记录。", "Changing your display name or username does not change your internal account ID, saved code, or submissions.")}</small>
          </label>
          <label>
            {t("个人简介", "Bio")}
            <textarea
              rows={3}
              maxLength={500}
              value={bio}
              disabled={changing}
              onChange={(e) => setBio(e.target.value)}
            />
          </label>
          <div className="settings-section">
            <h3>{t("公开资料", "Public visibility")}</h3>
            <p>
              {t(
                "昵称、头像和注册邮箱归属始终公开。帖子、回复、点赞、转发和关注关系会展示在你的主页。完整邮箱仅本人可见。",
                "Your display name, avatar, and registration affiliation are public. Posts, replies, likes, reposts, and follow connections appear on your profile. Your full email stays private.",
              )}
            </p>
          </div>
          {!user.is_test_user && <div className="public-affiliation-setting">
            <AffiliationBadge affiliation={user.affiliation} institution={user.institution} />
            <p>{t("注册邮箱对应的机构或邮箱域名固定公开，不能隐藏。邮箱域名仅表示邮箱归属，不代表已认证的学校或公司身份。", "Your registration institution or email domain is always public and cannot be hidden. An email domain identifies mailbox affiliation, not verified student or employment status.")}</p>
          </div>}
          <label>
            {t("国家／地区代码（选填）", "Country / region code (optional)")}
            <input
              maxLength={2}
              pattern="[A-Za-z]{2}|"
              value={country}
              disabled={changing}
              placeholder="US / CN / GB"
              onChange={(e) => setCountry(e.target.value.toUpperCase())}
            />
            <small>
              {t(
                "由你填写两位国家代码，不根据 IP 自动发布位置。",
                "Enter a two-letter country code. Your location is not published automatically from your IP.",
              )}
            </small>
          </label>
          <label className="social-checkbox">
            <input
              type="checkbox"
              checked={showCountry}
              disabled={changing}
              onChange={(e) => setShowCountry(e.target.checked)}
            />
            <span>{t("显示我的国家／地区", "Show my country or region")}</span>
          </label>
          <label className="social-checkbox">
            <input
              type="checkbox"
              checked={showProgress}
              disabled={changing}
              onChange={(e) => setShowProgress(e.target.checked)}
            />
            <span>
              {t("允许展示学习进度与勋章", "Allow learning progress and achievements to be public")}
              <small>
                {health?.mode === "hosted"
                  ? t(
                      "公开累计做题数、每日活跃记录、最近通过的题目及账号勋章。关闭后这些内容与置顶勋章仅自己可见；代码、草稿和详细判题报告始终私有。",
                      "Show your problem totals, daily activity, recent accepted problems, and account achievements. Turn this off to keep them and featured achievements visible only to you. Code, drafts, and detailed judge reports always stay private.",
                    )
                  : t(
                      "此开关控制账号判题统计与勋章的公开展示。本地浏览器练习记录不会自动上传；代码和详细报告始终私有。",
                      "Controls public statistics and achievements from saved account submissions. Browser-only practice is not uploaded automatically. Code and detailed reports always stay private.",
                    )}
              </small>
            </span>
          </label>
          <div className="form-footer">
            <button className="button" type="submit" disabled={changing}>
              {busy ? <span className="spinner" /> : <Check size={15} />}
              {t("保存设置", "Save changes")}
            </button>
          </div>
        </form>
        <aside className="account-aside">
          <section className="social-panel">
            {user.is_test_user ? (
              <>
                <h3>{t("内测账号", "Test account")}</h3>
                <TestAccountBadge isTestUser />
                <p className="private-email">
                  {user.login_username || user.username}
                </p>
                <p>
                  {t(
                    "这是管理员分配的登录账号，修改公开昵称或用户名不会改变它。访问码遗失时请联系管理员。",
                    "This is your administrator-issued login username. Changing your public name or profile handle does not change it. Contact an administrator if you lose your access code.",
                  )}
                </p>
              </>
            ) : (
              <>
                <h3>{t("邮箱已验证", "Email verified")}</h3>
                <p className="private-email">{user.email}</p>
                <AffiliationBadge affiliation={user.affiliation} institution={user.institution} />
                <p>
                  {user.institution ? t(
                    "此处的完整邮箱仅本人可见，注册邮箱对应的机构名称固定公开。",
                    "Only you can see your full email here. Your registration institution is always public.",
                  ) : t(
                    "此处的完整邮箱仅本人可见，注册邮箱域名公开。学校或公司名称尚未审核，域名不会显示机构认证标记。",
                    "Only you can see your full email here; your registration email domain is public. Your school or company name has not been reviewed, so the domain has no institutional verification badge.",
                  )}
                </p>
              </>
            )}
            <button
              className="button subtle logout-button"
              type="button"
              disabled={changing}
              onClick={() => void logout()}
            >
              <LogOut size={14} />
              {t("退出登录", "Sign out")}
            </button>
            <button className="text-button" type="button" disabled={busy} onClick={() => void logout(true)}>
              {t("退出所有设备", "Sign out on all devices")}
            </button>
            <p className="account-fineprint">{t("活跃使用自动保持登录；连续 90 天未使用需重新登录。退出会立即撤销服务器上的登录凭据。", "Active use keeps you signed in. Sign in again after 90 days of inactivity. Signing out immediately revokes the server session.")}</p>
          </section>
          <section className="social-panel">
            <h3>{t("公开预览", "Public preview")}</h3>
            <div className="profile-preview">
              <ProfileAvatarPreview user={{ ...user, display_name: name, avatar_color: color }} image={avatarImage} size={46} />
              <b>{name || t("你的昵称", "Your display name")}</b>
              <span>@{username}</span>
              <TestAccountBadge isTestUser={user.is_test_user} />
              {!user.is_test_user && (
                <AffiliationBadge affiliation={user.affiliation} institution={user.institution} />
              )}
              {showCountry && country && <span>{country}</span>}
              <p>{bio}</p>
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}

function ProfileAvatarPreview({ user, image, size }: { user: AccountUser; image: string | null | undefined; size: number }) {
  return typeof image === "string" ? <span className="social-avatar" style={{ width: size, height: size }} aria-label={user.display_name}>
    <img src={image} alt="" aria-hidden="true" />
  </span> : <Avatar user={{ ...user, avatar_url: image === null ? null : user.avatar_url }} size={size} />;
}
