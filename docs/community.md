# Community and institutional accounts

The community adds persistent posts, replies, likes, reposts, and follow relationships. Users have a public profile with activity tabs and lists of followers and followed accounts. Visitors can read public content; writing and following require sign-in. The default mode verifies approved institutional mailboxes. For a beta without real registration, explicitly enable [administrator-created test accounts](test-accounts.md).

The default **local development service** runs practice code on your machine and keeps drafts in the browser. The separate [hosted deployment](hosting.md) provides an isolated judge and private per-user cloud drafts. A deployment package is available; no public server has been provisioned by these implementation steps.

## Module discussions

Each practice workspace has a **Discuss** link. It opens that module's discussions, with a **Back to practice** link that preserves the private editor draft. New posts from this view carry the module's canonical ID; only the text entered in the composer is published. Code drafts, reports, and test artifacts are never attached automatically. General community posts remain available without a module association.

Module scope intersects board, keyword, following, and pagination filters. The module label appears on feed cards, post details, and profile activity, with links back to its discussions and exercise. Visitors can read threads; posting still requires sign-in. Existing posts keep a null association when the database upgrades. Threads for retired modules stay readable and do not redirect to a newly numbered exercise.

The collection API accepts `GET /api/community/posts?problem_id=architecture%2Fattention`. Creating a post accepts an optional canonical `problem_id` in the existing POST payload. The server validates it against the maintained catalog and derives public module metadata itself. Submitted titles, numbers, and author objects cannot spoof this context. Omit the parameter for the general feed.

## Public profiles and privacy

| Field or activity | Visibility |
| --- | --- |
| Display name, username, avatar | Always public; display name cannot be empty |
| Posts, replies, likes, reposts | Public on the author's profile |
| Followers and following | Public |
| Verified institution | Hidden by default; user controls its switch |
| Country or region | Optional and hidden by default; user controls its switch |
| Full email, provider identity, session data | Private; excluded from public API projections |
| Learning progress | Profile totals, activity calendar, and recent accepted problems follow the learning-progress visibility setting; the owner can always see their own statistics |
| Problem statistics | Public aggregate participants, solvers, full submissions, accepted/failed counts, and acceptance rate; no user identities, code, or individual reports |
| Test-account label | Always public for administrator-created test identities; no verified institution is claimed |

Avatars currently use generated initials and a configurable color. Uploading profile photos is not implemented. The country field is a user-entered ISO country or region code; the application does not look up, infer, or publish an IP-derived location. The backend uses the same public-user projection on profiles, posts, replies, and follow lists, so turning off institution visibility also removes it from those author cards.

The learning-progress preference controls public profile statistics. When disabled, the profile practice endpoint returns only `{"visible":false}` to other visitors; the signed-in owner can still view their dashboard. Code, drafts, detailed reports, and credentials are never included. Browser-only drafts and reports are not uploaded. Community content and follow relationships remain public regardless of this preference.

The profile dashboard counts distinct attempted and solved problems from full server-judged submissions, with difficulty totals, submission counts, a yearly activity calendar, and recent accepted problems. Daily activity follows the viewer's IANA time zone (UTC when omitted from the API); repeated submissions count separately but cumulative solved problems are deduplicated. Each day reports distinct attempted and accepted problems plus first-ever passes. The current streak includes a run ending today or yesterday; the longest streak is for the selected year. Partial-layer checks, mutation checks, and judge infrastructure failures do not count. Community received-like and received-repost totals exclude self-reactions.

## Configure real email verification

1. Create a Supabase project and enable the Email authentication provider, with new-user signup allowed. PaperLeet calls Supabase Auth on the server and verifies the returned user identity and confirmed email before creating a local account.
2. In Supabase's **Magic Link** email template, include `{{ .Token }}` so the recipient receives a code they can enter into PaperLeet. The default link-only template does not provide the code input flow. See [Supabase email OTP setup](https://supabase.com/docs/guides/auth/auth-email-passwordless) and [email templates](https://supabase.com/docs/guides/auth/auth-email-templates).
3. Configure custom SMTP in Supabase to send authentication messages to general users. Supabase's default email service restricts recipients to the project's team and is intended for testing. See [custom SMTP setup](https://supabase.com/docs/guides/auth/auth-smtp).
4. Review the eligible domains in [`institutions.yaml`](../institutions.yaml). Only exact listed domains can request a PaperLeet sign-in code. Verification proves control of an institutional mailbox, not current student or employee status.
5. Configure the environment in the same terminal that starts `make dev`:

```bash
cp .env.example .env
# Edit .env with your Supabase project URL and publishable key.
set -a
source .env
set +a
make dev
```

`.env` is ignored by Git. The application **does not automatically load it**; copying it alone will not enable sign-in. Restart the service after changing environment variables.

| Variable | Purpose |
| --- | --- |
| `PAPERLEET_AUTH_MODE` | `institutional` by default; `test` uses operator-issued login names and access codes, with no email provider |
| `SUPABASE_URL` | Supabase project URL, normally `https://PROJECT.supabase.co` |
| `SUPABASE_PUBLISHABLE_KEY` | Project publishable API key; the legacy `SUPABASE_ANON_KEY` is also accepted |
| `PAPERLEET_ADMIN_EMAILS` | Optional comma-separated institutional mailboxes granted local administrator role after successful verification |
| `PAPERLEET_SECURE_COOKIES` | `0` for local HTTP; `1` adds the Secure cookie attribute when an HTTPS deployment exists |

No Supabase service-role key is needed. Do not place secrets in `VITE_*` variables. The frontend talks to PaperLeet's `/api/account/` endpoints; it does not receive provider tokens. PaperLeet uses an opaque, HttpOnly session cookie with a seven-day lifetime and stores only the token's hash. Logging out revokes the current local session. Users cannot edit their verified institution or administrator role through profile settings.

If institutional configuration is missing, the sign-in screen explains that verification is unavailable. Public community browsing and local practice remain available. Test accounts require explicit `PAPERLEET_AUTH_MODE=test` and individual operator-issued access codes; they do not claim email verification and cannot authenticate in institutional mode. Email sending and verification use bounded provider timeouts and per-email request limits; the HTTP adapter also limits writes.

## Review institution requests

The initial allowlist contains `mit.edu` and `berkeley.edu`, with official sources recorded alongside their entries. Other schools and research institutions can submit a name, official website, and email domain from the sign-in page. Requests are saved as intake records; submitting a request does not activate an account or approve a domain.

Inspect the request queue from the repository root:

```bash
.venv/bin/python -m paperleet.accounts institution-requests
```

This command is read-only. It prints each request and `domain_approved`, derived from the current allowlist. Use `--database /path/to/community.sqlite3` to inspect a different local database. The request's stored `status` records its intake state; domain approval is controlled by the reviewed configuration.

For an approved request, a maintainer verifies that an official institution source documents the email domain, then adds the exact domain and its evidence URL to `institutions.yaml`. Review that change as a normal repository change and restart the service. Use a stable institution ID and names in both languages where available. A domain belongs to one institution; duplicate mappings are rejected. A listed parent domain does not approve its subdomains, and `.edu` suffixes have no special privilege. Do not add a personal email provider as an institution. Unknown domains remain unable to request a sign-in code until this review is complete.

## Data and maintenance

Accounts, private email identities, test credential hashes, sessions, institution requests, posts, replies, reactions, and follow relationships are stored in `community.sqlite3` under `PAPERLEET_DATA_DIR` (default `.paperleet/`). Local runtime data is ignored by Git; hosted data lives in a persistent Docker volume. Use the [backup commands](hosting.md#backup-and-restore) for online SQLite snapshots. The accounts and community services share the users table; Supabase Auth does not host these application tables.

The community supports author deletion of posts and replies, with administrator moderation enforced on the backend. Removing content also removes it from public activity results. Likes, reposts, and follows are idempotent operations; repeated requests do not inflate counts. Profile edits cannot change another user's profile or expose email through community responses.

Tests cover provider-response validation, domain gating, identity binding, session expiry and revocation, privacy projections, profile validation, and community ownership. Test-account checks exercise real local credentials, mode boundaries, reset and disable operations without an email provider. **No live verification email was sent as part of implementation validation.** Institutional mode still requires a configured Supabase project and an approved mailbox to verify actual email delivery; test mode does not send mail.
