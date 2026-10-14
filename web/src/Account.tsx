import { useEffect, useState } from "react";
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
  InstitutionBadge,
  SocialNotice,
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
          loginError={typeof location === "undefined" ? undefined : organizationLoginError(decodeProblemHash(location.hash))} />
      ) : (
        <SignIn enabled={capabilities?.auth_available === true} workEmailRegistration={capabilities?.work_email_registration === true} />
      )}
    </main>
  );
}

export function OrganizationSignIn({ providers, returnTo, loginError }: {
  providers: NonNullable<AccountCapabilities["oauth_providers"]>; returnTo?: string; loginError?: string;
}) {
  const { t } = useI18n();
  const [busy, setBusy] = useState<string>(), [error, setError] = useState("");
  const failures: Record<string, string> = {
    identity: t("未能确认学校／企业组织身份。请使用 Google Workspace 或 Microsoft 工作／学校账号；若仍失败，请联系管理员。", "We could not confirm your organization identity. Use a Google Workspace or Microsoft work/school account. Contact an administrator if this continues."),
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
  return <div className="account-grid">
    <section className="social-panel login-card">
      <span className="social-eyebrow">PAPERLEET</span>
      <h1>{t("学校／企业账号登录", "School or company sign-in")}</h1>
      <p className="social-description">{t("使用学校或公司的组织账号登录，首次登录将创建 PaperLeet 账号，无需接收本站验证码。", "Sign in with your school or company organization account. Your first sign-in creates a PaperLeet account without an emailed code from us.")}</p>
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
      <p className="social-description">{t("请在登录页面选择组织账号。普通 Gmail、Outlook、Hotmail 个人账号不符合当前注册规则。", "Choose your organization account. Personal Gmail, Outlook and Hotmail accounts are not eligible under the current registration policy.")}</p>
    </section>
    <aside className="social-panel login-card">
      <h2>{t("哪些邮箱可以使用？", "Which accounts work?")}</h2>
      <p className="social-description">{t("Google Workspace 和 Microsoft 365／Entra 管理的学校、公司账号。部分机构可能要求管理员批准应用。", "School and company accounts managed by Google Workspace or Microsoft 365/Entra. Some organizations require administrator approval.")}</p>
      <p className="social-description">{t("其他邮件系统暂不支持这两个登录入口。邮箱地址不会显示在公开主页，组织账号也不代表学校或雇主对个人身份的背书。", "Other email systems are not covered by these two sign-in options. Your email stays private. An organization account is not an endorsement by a school or employer.")}</p>
    </aside>
  </div>;
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
                "Test accounts are created by an administrator. Display names and avatars are always public. Local-mode practice stays in this browser.",
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
export function SignIn({ enabled, workEmailRegistration = false }: { enabled: boolean; workEmailRegistration?: boolean }) {
  const { t, locale } = useI18n();
  const { setUser, capabilities } = useAccount();
  const [email, setEmail] = useState(""),
    [code, setCode] = useState(""),
    [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const domain = email.trim().toLowerCase().split("@")[1];
  const institution = capabilities?.institutions?.find((i) =>
    i.domains.includes(domain),
  );
  async function sendCode() {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await socialApi("account/send-code", { email: email.trim() });
      setSent(true);
      setMessage(
        t(
          "验证码已发送，请检查学校或公司邮箱。",
          "A verification code has been sent to your school or company email.",
        ),
      );
    } catch (e) {
      setError(errorMessage(e));
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
            "用学校或公司邮箱注册／登录，与大家讨论论文和实现。",
            "Register or sign in with your school or company email to discuss papers and implementations.",
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
            {t("开放学校和公司邮箱注册，无需邀请码。Gmail、QQ、Outlook 等个人邮箱暂不支持；收到并验证邮件验证码后才会创建账户。", "School and company email registration is open without invitations. Personal email services such as Gmail, QQ and Outlook are excluded. Your account is created after you verify the emailed code.")}
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
              placeholder="you@university.edu"
              onChange={(e) => setEmail(e.target.value)}
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
          <button className="button" type="submit" disabled={busy || !enabled}>
            {busy ? (
              <span className="spinner" />
            ) : sent ? (
              <ShieldCheck size={15} />
            ) : (
              <Mail size={15} />
            )}
            {sent
              ? t("验证并登录", "Verify and sign in")
              : t("发送验证码", "Send verification code")}
          </button>
          {sent && (
            <div className="form-inline-actions">
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setSent(false);
                  setCode("");
                  setMessage("");
                }}
              >
                {t("更换邮箱", "Change email")}
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void sendCode()}
              >
                {t("重新发送", "Resend code")}
              </button>
            </div>
          )}
        </form>
        <p className="account-fineprint">
          {workEmailRegistration ? t(
            "昵称和头像始终公开，完整邮箱保持私密。机构名称经域名审核后才会显示认证标记。",
            "Your display name and avatar are always public; your full email stays private. Institution badges require a reviewed domain.",
          ) : t(
            "只有通过审核的机构邮箱可以注册。昵称和头像始终公开，机构和国家／地区由你选择是否展示。",
            "Registration is limited to approved institutional email domains. Your display name and avatar are always public. You choose whether to display your institution and country or region.",
          )}
        </p>
      </section>
      <aside className="account-aside">
        <div className="social-panel">
          <ShieldCheck size={27} className="accent-icon" />
          <h3>{t("学校／公司邮箱验证", "School / company email verification")}</h3>
          <p>
            {t(
              "验证说明你拥有这个机构邮箱，不代表在读或在职身份认证。完整邮箱不会出现在公开主页。",
              "Verification confirms access to an institutional mailbox. It does not certify current student or employment status. Your full email address is never shown on your public profile.",
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
          "可以先验证邮箱注册。提交学校或公司信息，经审核后可在个人资料中展示机构名称。",
          "You can register by verifying your email first. Submit your school or company details for review to display an institution name on your profile.",
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
  const { t } = useI18n();
  const { setUser, health } = useAccount();
  const [name, setName] = useState(user.display_name),
    [username, setUsername] = useState(user.username),
    [color, setColor] = useState(user.avatar_color),
    [bio, setBio] = useState(user.bio || "");
  const [country, setCountry] = useState(user.country || ""),
    [showInstitution, setShowInstitution] = useState(user.show_institution),
    [showCountry, setShowCountry] = useState(user.show_country),
    [showProgress, setShowProgress] = useState(user.show_progress);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  useEffect(() => {
    setMessage("");
  }, [
    name,
    username,
    color,
    bio,
    country,
    showInstitution,
    showCountry,
    showProgress,
  ]);
  async function save() {
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<{ user: AccountUser }>("account/profile", {
        display_name: name.trim(),
        username: username.trim(),
        avatar_color: color,
        bio,
        country: country.trim().toUpperCase(),
        show_institution: showInstitution,
        show_country: showCountry,
        show_progress: showProgress,
      });
      setUser(data.user);
      setMessage(t("个人资料已保存。", "Your profile has been saved."));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function logout() {
    setBusy(true);
    setError("");
    try {
      await socialApi("account/logout", {});
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
          <span className="social-eyebrow">YOUR ACCOUNT</span>
          <h1>{t("个人资料与隐私", "Profile & privacy")}</h1>
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
          {message && <SocialNotice>{message}</SocialNotice>}
          <div className="settings-avatar">
            <Avatar
              user={{ display_name: name, avatar_color: color }}
              size={64}
            />
            <div>
              <h3>{t("公开头像", "Public avatar")}</h3>
              <div className="avatar-palette">
                {COLORS.map((c) => (
                  <button
                    key={c}
                    type="button"
                    aria-label={`${t("头像颜色", "Avatar color")} ${c}`}
                    aria-pressed={color === c}
                    style={{ backgroundColor: c }}
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
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            {t("用户名", "Username")}
            <input
              required
              minLength={3}
              maxLength={24}
              pattern="[a-z0-9_]{3,24}"
              value={username}
              onChange={(e) => setUsername(e.target.value.toLowerCase())}
            />
            <small>
              {t(
                "3–24 位小写字母、数字或下划线，用于个人主页链接。",
                "3–24 lowercase letters, numbers, or underscores. Used in your profile link.",
              )}
            </small>
          </label>
          <label>
            {t("个人简介", "Bio")}
            <textarea
              rows={3}
              maxLength={500}
              value={bio}
              onChange={(e) => setBio(e.target.value)}
            />
          </label>
          <div className="settings-section">
            <h3>{t("公开资料", "Public visibility")}</h3>
            <p>
              {t(
                "昵称和头像始终公开。帖子、回复、点赞、转发和关注关系会展示在你的主页。",
                "Your display name and avatar are always public. Posts, replies, likes, reposts, and follow connections appear on your profile.",
              )}
            </p>
          </div>
          {!user.is_test_user && user.institution && (
            <label className="social-checkbox">
              <input
                type="checkbox"
                checked={showInstitution}
                onChange={(e) => setShowInstitution(e.target.checked)}
              />
              <span>
                {t("显示已验证机构", "Show my verified institution")}
                <small>{user.institution?.name}</small>
              </span>
            </label>
          )}
          <label>
            {t("国家／地区代码（选填）", "Country / region code (optional)")}
            <input
              maxLength={2}
              pattern="[A-Za-z]{2}|"
              value={country}
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
              onChange={(e) => setShowCountry(e.target.checked)}
            />
            <span>{t("显示我的国家／地区", "Show my country or region")}</span>
          </label>
          <label className="social-checkbox">
            <input
              type="checkbox"
              checked={showProgress}
              onChange={(e) => setShowProgress(e.target.checked)}
            />
            <span>
              {t("允许展示学习进度", "Allow learning progress to be public")}
              <small>
                {health?.mode === "hosted"
                  ? t(
                      "公开累计做题数、每日活跃记录和最近通过的题目。关闭后仅自己可见；代码、草稿和详细判题报告始终私有。",
                      "Show your problem totals, daily activity, and recent accepted problems. Turn this off to keep these visible only to you. Code, drafts, and detailed judge reports always stay private.",
                    )
                  : t(
                      "此开关控制账号中已保存判题记录的公开统计。本地模式的浏览器练习记录不会自动上传；代码和详细报告始终私有。",
                      "Controls public statistics from saved account submissions. Browser-only practice is not uploaded automatically. Code and detailed reports always stay private.",
                    )}
              </small>
            </span>
          </label>
          <div className="form-footer">
            <button className="button" type="submit" disabled={busy}>
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
                <InstitutionBadge institution={user.institution} />
                <p>
                  {user.institution ? t(
                    "此处的完整邮箱仅本人可见。机构名称是否公开由左侧开关控制。",
                    "Only you can see your full email here. Use the visibility setting to control whether your institution appears publicly.",
                  ) : t(
                    "此处的完整邮箱仅本人可见。邮箱归属已验证，学校或公司名称尚未审核。",
                    "Only you can see your full email here. Mailbox ownership is verified; your school or company name has not been reviewed.",
                  )}
                </p>
              </>
            )}
            <button
              className="button subtle logout-button"
              type="button"
              disabled={busy}
              onClick={() => void logout()}
            >
              <LogOut size={14} />
              {t("退出登录", "Sign out")}
            </button>
          </section>
          <section className="social-panel">
            <h3>{t("公开预览", "Public preview")}</h3>
            <div className="profile-preview">
              <Avatar
                user={{ display_name: name, avatar_color: color }}
                size={46}
              />
              <b>{name || t("你的昵称", "Your display name")}</b>
              <span>@{username}</span>
              <TestAccountBadge isTestUser={user.is_test_user} />
              {!user.is_test_user && showInstitution && (
                <InstitutionBadge institution={user.institution} />
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
