# Host an invitation-only beta

[English](hosting.md) | [简体中文](hosting.zh-CN.md)

This deployment serves the paper library, accounts, community, and the 60 maintained practice modules. You still need a server and domain. For a beta without real registration, use [test accounts](test-accounts.md): set `PAPERLEET_AUTH_MODE=test` and create a login name and access code for each tester. Supabase and email configuration are only required for institutional sign-in.

To run the beta from your own computer before renting a server, follow the [Cloudflare Tunnel guide](cloudflare-tunnel.md). That profile keeps the same hosted account and judging boundaries, with a private HTTP proxy behind the tunnel instead of public server ports.

## Server and architecture

Start with a Linux server with **2 vCPUs and 4 GB RAM**, rootful Docker Engine, the Docker Compose plugin, and persistent disk space for images, application data, and backups. The default judge runs one job at a time, with a 2 GB memory limit and two CPU cores. No GPU is needed. Increase resources before raising judge concurrency.

Point your domain's DNS record at the server. Allow inbound TCP ports **80 and 443** so Caddy can obtain certificates and serve HTTPS; keep your administrative SSH access separately controlled. Remove a stale IPv6 DNS record if the server does not serve that address. Do not publish ports 8765 or 8766.

```text
Browser → Caddy (HTTPS + static frontend) → API → private judge manager
                                           │           │
                                      SQLite volume    └→ temporary offline worker
```

Only Caddy publishes ports. The API holds accounts and application data; it has no Docker socket. The internal judge manager holds the Docker socket and starts disposable containers with no host directory mounts, network, or application credentials. The manager's socket access effectively grants control of the Docker host: use a dedicated server for this beta and keep its port and token private. Containers share the host kernel; this setup is intended for invited users rather than a hostile public execution service.

## Configure access and email

Choose `test` authentication for administrator-created beta accounts, or `institutional` (the default) for verified email sign-in. The [test-account guide](test-accounts.md) replaces the email setup below; server, domain, storage and judging configuration remain the same. In test mode, leave Supabase and invited/admin email settings empty; create accounts and assign administrator roles through the maintenance command instead.

Clone the repository on the server, then work from its root:

```sh
git clone https://github.com/Hongyang-Du/PaperLeet.git
cd PaperLeet
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
openssl rand -hex 32
```

Edit `deploy/.env`. Put the generated token in `PAPERLEET_JUDGE_TOKEN`; do not leave the example value in place.

| Setting | Value |
| --- | --- |
| `PAPERLEET_DOMAIN` | Your DNS name, such as `paperleet.yourdomain.com`; no scheme or path |
| `PAPERLEET_AUTH_MODE` | `institutional` by default; `test` enables administrator-created accounts without email |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_PUBLISHABLE_KEY` | The project's publishable key, or its legacy anon key; never a service-role key |
| `PAPERLEET_INVITE_EMAILS` | Comma-separated **exact institutional email addresses** admitted to the beta |
| `PAPERLEET_ADMIN_EMAILS` | Optional invited institutional addresses that receive administrator privileges |
| `PAPERLEET_JUDGE_TOKEN` | The random token shared only by the API and judge manager |

Every invited email must also match an approved **exact domain** in [`institutions.yaml`](../institutions.yaml). Invitations and institution approval are separate checks. Adding an address does not approve its school; adding a school does not invite everyone at that school. Initially `mit.edu` and `berkeley.edu` are approved. Review official institution evidence before adding more domains, including specific student subdomains. An institution-request form only creates a request for review.

In Supabase, enable Email authentication and signup. Add `{{ .Token }}` to the **Magic Link** email template so users receive a code they can enter into PaperLeet. Configure custom SMTP for delivery to invited users; the default Supabase sender restricts recipients to project-team addresses. See [email OTP](https://supabase.com/docs/guides/auth/auth-email-passwordless), [email templates](https://supabase.com/docs/guides/auth/auth-email-templates), and [SMTP setup](https://supabase.com/docs/guides/auth/auth-smtp).

Mailbox verification establishes control of an institutional email address; it does not establish current student or employee status. Email and session details stay private. Display names and avatars are always public; institution and country visibility are separate user settings. The country is user-entered, not inferred from IP.

## Build, start, and check

```sh
./deploy/manage.sh build
./deploy/manage.sh up
./deploy/manage.sh status
```

The build creates the CPU worker image and the API, judge-manager, and web images. Startup waits for API health, including judge readiness. Caddy may still need time to obtain the certificate after DNS becomes reachable.

In test-account mode, create accounts with `./deploy/manage.sh test-users create tester1 --name "Tester 1"`, then verify login and judging using the issued access code. The mailbox checks below apply to institutional mode only.

Check the real domain:

```sh
curl --fail https://paperleet.yourdomain.com/api/health
./deploy/manage.sh logs
```

The health response should contain `"mode":"hosted"`, `"auth_required":true`, and `"judge":{"ready":true,...}`. Confirm the certificate, then test a real invited mailbox end to end: receive a code, sign in, run a module, and reopen its draft from another browser after signing into the same account. Test an uninvited address and a wrong-domain address too; neither should gain access. Do not use mocked accounts as evidence that SMTP works.

The website submits asynchronous jobs to a SQLite FIFO queue and polls short status requests. The API executes one queued job at a time, with at most eight active/queued jobs in total and one per account. The UI shows the number of jobs ahead and allows cancelling an unstarted job. Refreshing the same browser tab reconnects to its job; editing during a run does not change the submitted source. Temporary polling failures reconnect without re-executing code. A full queue still returns a clear capacity error; drafts are preserved. Queued jobs have a 20-minute wait limit, and finished job status is retained for 24 hours before cleanup. Keep one API instance for this queue.

Jobs are stored in `community.sqlite3` and included in the normal database backup. After restart, unstarted jobs resume. A previously running job is marked interrupted rather than automatically replayed, because its result may already have been stored; the UI asks the user to submit again. Hosted practice history remains in the existing private run tables. The legacy synchronous `/api/test` endpoint remains compatible; the website uses `/api/judge/jobs`.

Hosted judging supports the **60 maintained modules** only. It never falls back to running submitted Python inside the API container. Test assets and references are available inside workers, so accepted results are learning feedback, not cheat-proof competition scores.

Paper proposals are saved privately for their owner and administrators. Generation, validation, review, and publication remain an **offline maintainer workflow** during the beta. Generated community exercises are not accepted by the hosted judge. Profiles show community activity and, when allowed by the progress visibility setting, problem totals, daily activity, and recent accepted problems. With that setting off, only the account owner can see their learning statistics. Signed-in practice drafts and detailed run history remain private on the server. The public read-only `/api/practice/stats` endpoint exposes aggregate per-problem counts derived from full server-judged submissions: distinct participants/solvers, total/accepted/failed submissions, and acceptance rate. Partial checks and judge-service failures do not contribute; existing stored runs are included. The endpoint does not return user identities, source code, or reports.

## Persistent data and updates

The `app_data` named volume stores the application data directory, including `community.sqlite3`, practice drafts and run history, and paper proposals. Caddy's two named volumes preserve certificates and configuration state. Rebuilding images or using `./deploy/manage.sh down` preserves these volumes. **Do not use `docker compose down -v`** to update or restart the service.

```sh
./deploy/manage.sh backup
git pull --ff-only
./deploy/manage.sh build
./deploy/manage.sh up
```

Rebuilding is also required after changing the tracked institution allowlist. Editing `deploy/.env` and running `up` recreates services with the changed settings. Removing an email from the invite list prevents that account from using hosted sessions after the API is recreated. Revoking an institution domain also requires rebuilding and restarting the API.

The current persistence layer is single-instance SQLite. Configuring Supabase Auth does not upload application tables to Supabase PostgreSQL. Keep one API instance attached to the local persistent volume; a managed PostgreSQL migration is separate work.

## Backup and restore

```sh
./deploy/manage.sh backup
```

This writes a private `deploy/backups/paperleet-TIMESTAMP.tar.gz` archive using SQLite's online-backup mechanism. Copy completed archives to private storage outside the server and test restoration periodically. They contain private account and practice data. Backups and `deploy/.env` are excluded from Git and image build contexts. Keep a separate private copy of `deploy/.env`; the archive does **not** contain deployment settings, Supabase project configuration, SMTP configuration, or Caddy's certificate volumes. Institution configuration lives in the tracked `institutions.yaml`; retain the corresponding repository revision.

Restore only into a fresh empty data volume. The command refuses to overwrite existing data. On a replacement server, configure `deploy/.env`, build the images, and restore **before** starting the API:

```sh
./deploy/manage.sh build
./deploy/manage.sh restore /absolute/path/to/paperleet-backup.tar.gz
./deploy/manage.sh up
```

To rehearse a restore on the same server without deleting the original volumes, use a new Compose project name consistently. Stop the old deployment first to release ports 80 and 443:

```sh
./deploy/manage.sh down
COMPOSE_PROJECT_NAME=paperleet-restore ./deploy/manage.sh restore /absolute/path/to/paperleet-backup.tar.gz
COMPOSE_PROJECT_NAME=paperleet-restore ./deploy/manage.sh up
COMPOSE_PROJECT_NAME=paperleet-restore ./deploy/manage.sh status
```

The original project's data remains available. To return to it, stop the restore project with the same `COMPOSE_PROJECT_NAME` prefix, then run `./deploy/manage.sh up` without that prefix. Do not delete either project's volumes until its data is no longer needed.

## Validation before inviting users

Repository checks cover hosted origin and invite restrictions, account privacy, private practice ownership, report integrity, container limits, and deployment configuration. A built worker can be checked on the target server with:

```sh
PAPERLEET_DOCKER_SMOKE=1 .venv/bin/python -m pytest tests/test_hosted_judge.py -k live_docker -q
PAPERLEET_TEST_DEPLOYMENT=1 .venv/bin/python -m pytest tests/test_deployment_live.py -q
```

These optional development commands require the repository's Python test environment; production containers do not need a host Python installation. The first runs all ten references, a failing starter, isolation probes, and a timeout-cleanup check. The second requires all four deployment images to be built; it creates a temporary Compose project with fake credentials and loopback HTTPS to test private practice, real judging, restart persistence, backup, and empty-volume restore, then removes its test resources. It does not send email or publish a public site. Actual DNS, publicly trusted TLS, SMTP delivery, disk backup storage, and restoration must also be checked on your deployment.
