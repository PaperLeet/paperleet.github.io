import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";
const server = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
});
try {
  const { TestSignIn, SignIn, ProfileSettings } =
    await server.ssrLoadModule("/src/Account.tsx");
  const { PersonLink, AccountNav } = await server.ssrLoadModule("/src/social.tsx");
  const { signInHref, accountReturnHref, isAccountRoute } = await server.ssrLoadModule("/src/accountNavigation.ts");
  const { decodeProblemHash } = await server.ssrLoadModule("/src/reports.ts");
  for (const route of ["", "22", "architecture/attention", "finetuning/lora", "community", "community/problem/architecture/attention", "community/0123456789abcdef0123456789abcdef", "u/researcher", "submit", "papers/transformer", "collections/llm-basics"]) {
    const href = signInHref(route);
    const accountRoute = decodeProblemHash(href);
    assert(isAccountRoute(accountRoute));
    assert.equal(accountReturnHref(accountRoute), `#/${route}`);
  }
  assert.equal(accountReturnHref(decodeProblemHash(signInHref("#/community/problem/architecture%2Fattention"))), "#/community/problem/architecture/attention");
  for (const route of ["https://evil.test/", "//evil.test", "javascript:alert(1)", "account", "account/continue/community", "community/problem/../attention", "community/problem/architecture/attention?x=y", "#/%2F%2Fevil.test", "#/%ZZ", "#/account/continue/submit", "#/community%252Fproblem%252Farchitecture%252Fattention"]) {
    assert.equal(signInHref(route), "#/account", `Unsafe/recursive continuation: ${route}`);
  }
  assert.equal(accountReturnHref("account/continue/https://evil.test"), undefined);
  assert.equal(accountReturnHref("account/continue/account/continue/community"), undefined);
  assert.equal(accountReturnHref("account/continue/%2F%2Fevil.test"), undefined);
  const nav = renderToString(React.createElement(AccountNav, { returnTo: "architecture/attention" }));
  assert.equal(nav, "", "Standalone local practice must not require an account service");
  const login = renderToString(
    React.createElement(TestSignIn, { enabled: true }),
  );
  assert(
    login.includes('type="password"'),
    "Access codes must use a masked input",
  );
  assert(
    login.includes('autoComplete="username"'),
    "Test login must request the issued account name",
  );
  assert(
    !login.includes('type="email"'),
    "Test mode must not require an institutional email",
  );
  assert(
    !login.includes("one-time-code"),
    "Test mode must not request an emailed OTP",
  );
  assert(
    !login.includes("申请收录"),
    "Institution review is not part of test login",
  );
  const institutional = renderToString(
    React.createElement(SignIn, { enabled: true }),
  );
  assert(
    institutional.includes('type="email"'),
    "Institutional sign-in must remain available in its own mode",
  );
  assert(
    institutional.includes("发送验证码"),
    "Institutional sign-in must keep email verification",
  );
  const openRegistration = renderToString(React.createElement(SignIn, { enabled: true, workEmailRegistration: true }));
  assert(openRegistration.includes("学校／公司邮箱"));
  assert(openRegistration.includes("无需邀请码"));
  assert(openRegistration.includes("Gmail、QQ、Outlook"));
  assert(!openRegistration.includes("只有通过审核的机构邮箱可以注册"));
  assert(!openRegistration.includes("管理员审核域名后即可使用邮箱验证"));
  assert(openRegistration.includes("不会出现在公开主页"));
  const testUser = {
    id: "qa-test",
    username: "public_handle",
    login_username: "issued_login",
    display_name: "QA Test User",
    avatar_color: "#4f7c73",
    bio: "",
    created_at: "2026-10-13T00:00:00Z",
    is_test_user: true,
    email: "",
    role: "member",
    institution: null,
    institution_verified: false,
    country: "",
    show_institution: false,
    show_country: false,
    show_progress: false,
  };
  const settings = renderToString(
    React.createElement(ProfileSettings, { user: testUser }),
  );
  assert(
    settings.includes("issued_login"),
    "Settings must display the stable login username",
  );
  assert(
    settings.includes("public_handle"),
    "The public handle must remain distinct from login credentials",
  );
  assert(
    !settings.includes("已验证的账户"),
    "Test accounts must not claim verified institutional identity",
  );
  assert(
    !settings.includes("显示已验证机构"),
    "A null test institution must not have a visibility control",
  );
  const publicLink = renderToString(
    React.createElement(PersonLink, { user: testUser }),
  );
  assert(
    publicLink.includes("内测账号"),
    "Public test identities must be labeled",
  );
  assert(
    !publicLink.includes("issued_login"),
    "The private login name must not appear in public identity links",
  );
  const companyUser = { ...testUser, is_test_user: false, email_verified: true, email: "developer@company.example", login_username: undefined };
  const companySettings = renderToString(React.createElement(ProfileSettings, { user: companyUser }));
  assert(companySettings.includes("邮箱已验证"));
  assert(companySettings.includes("学校或公司名称尚未审核"));
  assert(!companySettings.includes("显示已验证机构"));
  assert(!renderToString(React.createElement(PersonLink, { user: companyUser })).includes(companyUser.email));
  console.log(
    "PASS test-account login, masked access codes, separate public identity and no institutional verification claims",
  );
} finally {
  await server.close();
}
