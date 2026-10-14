# 部署邀请制内测版

[English](hosting.md) | [简体中文](hosting.zh-CN.md)

这套部署提供论文题库、账号、社区，以及当前 60 个正式维护模块的在线判题。上线需要服务器和域名。若内测暂不做真实注册，可以使用[测试账号](test-accounts.zh-CN.md)：设置 `PAPERLEET_AUTH_MODE=test`，为每位参与者创建登录名和访问码即可。只有机构邮箱登录模式才需要 Supabase 和邮件配置。

租服务器之前，也可以按 [Cloudflare Tunnel 指南](cloudflare-tunnel.zh-CN.md)在自己的电脑上开放内测。该配置保留相同的 hosted 账号和隔离判题机制，使用隧道后的内部 HTTP 代理，无需开放公网入站端口。

## 服务器与架构

建议从 **2 核 CPU、4 GB 内存**的 Linux 服务器开始，安装 rootful Docker Engine 和 Docker Compose 插件，并为镜像、应用数据和备份保留持久磁盘空间。默认一次运行一个判题任务，每个任务限制 2 GB 内存和两个 CPU 核，不需要 GPU。提高并发前先增加服务器资源。

将域名 DNS 指向服务器，开放入站 TCP **80 和 443** 端口，让 Caddy 申请证书并提供 HTTPS；SSH 管理权限单独控制。如果服务器不支持对应 IPv6 地址，删除过期的 IPv6 DNS 记录。不要把 8765 或 8766 端口暴露到公网。

```text
浏览器 → Caddy（HTTPS 和静态网页）→ API → 内部判题管理服务
                                    │              │
                               SQLite 数据卷       └→ 临时离线判题容器
```

只有 Caddy 发布公网端口。API 保存账号和应用数据，不持有 Docker socket。内部判题管理服务使用 Docker socket 创建临时容器；这些容器没有宿主机目录挂载、网络或应用凭据。管理服务的 socket 权限实际等同于控制 Docker 宿主机，因此内测应使用专门的服务器，并保护其端口和令牌。容器共享宿主机内核，这套方案面向受邀用户内测，不是对任意敌对代码开放的公共执行服务。

## 配置邀请与邮箱验证

使用管理员创建的内测账号时选择 `test`；使用真实邮箱验证时选择 `institutional`（默认值）。[测试账号说明](test-accounts.zh-CN.md)可以替代下面的邮箱配置步骤，服务器、域名、存储和判题配置不变。测试模式下可以留空 Supabase、邀请邮箱和管理员邮箱，通过维护命令创建账号并授予管理员角色。

在服务器上克隆仓库，后续命令均从仓库根目录执行：

```sh
git clone https://github.com/Hongyang-Du/PaperLeet.git
cd PaperLeet
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
openssl rand -hex 32
```

编辑 `deploy/.env`，把生成的随机令牌填入 `PAPERLEET_JUDGE_TOKEN`，不要保留示例值。

| 配置 | 内容 |
| --- | --- |
| `PAPERLEET_DOMAIN` | 实际域名，如 `paperleet.yourdomain.com`，不带协议或路径 |
| `PAPERLEET_AUTH_MODE` | 默认为 `institutional`；设为 `test` 时使用管理员创建的测试账号，无需邮箱 |
| `SUPABASE_URL` | Supabase 项目 URL |
| `SUPABASE_PUBLISHABLE_KEY` | 项目的 publishable key，或旧版 anon key；不要使用 service-role key |
| `PAPERLEET_INVITE_EMAILS` | 允许参加内测的**完整机构邮箱地址**，以逗号分隔 |
| `PAPERLEET_ADMIN_EMAILS` | 可选：获得管理员权限的已受邀机构邮箱 |
| `PAPERLEET_JUDGE_TOKEN` | 仅由 API 与内部判题管理服务共享的随机令牌 |

受邀邮箱还必须匹配 [`institutions.yaml`](../institutions.yaml) 中审核通过的**精确邮箱域名**。邀请和机构审核是两个独立条件：填写邮箱不会自动批准学校；批准学校也不会邀请整个学校。初始批准的是 `mit.edu` 和 `berkeley.edu`。新增域名或学生邮箱子域名前，要核实机构官方资料。用户提交机构申请只会进入待审核队列。

在 Supabase 中开启 Email 登录和注册，在 **Magic Link** 邮件模板中加入 `{{ .Token }}`，让用户收到可以在 PaperLeet 输入的验证码。给真实受邀用户发信前配置自定义 SMTP；Supabase 默认发信服务只允许向项目团队邮箱发送。参考 [邮箱 OTP](https://supabase.com/docs/guides/auth/auth-email-passwordless)、[邮件模板](https://supabase.com/docs/guides/auth/auth-email-templates)、[SMTP 配置](https://supabase.com/docs/guides/auth/auth-smtp)。

邮箱验证证明用户能控制机构邮箱，不直接证明当前在读或在职。邮箱和会话数据保持私有；昵称与头像始终公开，机构和国家分别由用户选择是否显示。国家由用户填写，不从 IP 推断。

## 构建、启动与检查

```sh
./deploy/manage.sh build
./deploy/manage.sh up
./deploy/manage.sh status
```

构建命令生成 CPU 判题 worker、API、判题管理服务和网页镜像。启动会等待 API 健康检查通过，其中包括判题服务就绪。DNS 生效后，Caddy 申请证书仍可能需要一些时间。

测试账号模式下，执行 `./deploy/manage.sh test-users create tester1 --name "测试用户 1"` 创建账号，再使用返回的访问码验证登录和答题。下面的真实邮箱检查仅适用于机构邮箱模式。

检查真实域名：

```sh
curl --fail https://paperleet.yourdomain.com/api/health
./deploy/manage.sh logs
```

健康响应应包含 `"mode":"hosted"`、`"auth_required":true` 和 `"judge":{"ready":true,...}`。确认证书可用，然后用真实受邀邮箱完整测试：收验证码、登录、运行一道题，再从另一浏览器登录同一账号并打开草稿。同时测试一个未受邀邮箱和一个未批准机构域名，二者均不应获得使用权限。模拟账号测试不能证明 SMTP 已经可以实际发信。

网站提交异步任务到 SQLite FIFO 队列，并用短请求查询状态。API 一次执行一个排队任务，运行和等待中的任务合计最多 8 个，每个账号最多一个。页面显示前方任务数，可以取消尚未开始的任务；刷新同一浏览器标签页会恢复查看。判题过程中继续编辑不会改变本次提交的代码，短暂断网后只恢复查询，不重复执行。队列满时仍会明确提示容量已满，草稿保留。排队最长等待 20 分钟，终态任务保留 24 小时后可被清理。此队列只支持单 API 实例。

任务存储在 `community.sqlite3`，随正常数据库备份一起保存。服务重启后恢复尚未开始的任务；之前正在执行的任务标为中断，不自动重跑，因为结果可能已经存入练习历史，页面会提示重新提交。账号的私密判题历史继续使用原有存储。旧版同步 `/api/test` 接口保持兼容，网站改用 `/api/judge/jobs`。

在线判题目前只支持**正式维护的 60 个模块**，不会降级为在 API 容器内执行用户 Python。worker 可以读取测试和参考实现，因此通过结果用于学习反馈，不作为防作弊竞赛成绩。

论文提案会保存为仅投稿者和管理员可见的记录。内测期间，生成、验证、审核和发布仍由维护者**离线处理**；生成的社区题目暂不进入在线判题。个人主页展示社区动态，并按学习进度展示开关提供做题汇总、每日活跃日历和最近通过记录；关闭开关后仅本人可见。登录后的练习草稿与详细运行记录会私有地保存在服务器。公开只读接口 `/api/practice/stats` 从服务器的完整判题记录汇总每道题的做题人数、通过人数、提交总数、通过／未通过次数和通过率；局部测试和判题服务故障不计入，已有记录会自动纳入统计。该题目统计接口不返回用户身份、代码或个人报告。

## 持久数据与更新

`app_data` 命名数据卷保存应用数据目录，包括 `community.sqlite3`、练习草稿、运行记录和论文提案。Caddy 的两个命名数据卷保存证书与配置状态。重新构建镜像或执行 `./deploy/manage.sh down` 都会保留这些数据卷。**不要用 `docker compose down -v` 更新或重启服务。**

```sh
./deploy/manage.sh backup
git pull --ff-only
./deploy/manage.sh build
./deploy/manage.sh up
```

修改受 Git 管理的机构域名名单后，也需要重新构建。修改 `deploy/.env` 后执行 `up`，会使用新配置重新创建服务。从邀请名单移除邮箱后，在 API 重建运行配置后，该用户将不能继续使用线上会话；撤销机构域名还需要重新构建并重启 API。

当前持久层是单实例 SQLite。配置 Supabase Auth 不会把应用数据自动上传到 Supabase PostgreSQL。保留一个 API 实例和本地持久数据卷；迁移到托管 PostgreSQL 需要另做数据层适配。

## 备份与恢复

```sh
./deploy/manage.sh backup
```

命令使用 SQLite 在线备份机制，生成私有的 `deploy/backups/paperleet-TIMESTAMP.tar.gz` 文件。将完整备份复制到服务器之外的私有存储，并定期测试恢复。备份含有私人账号和练习数据。备份目录与 `deploy/.env` 均已排除在 Git 和镜像构建上下文之外。

另外私下保存 `deploy/.env`。备份包**不包含**部署配置、Supabase 项目配置、SMTP 配置或 Caddy 证书数据卷。机构配置位于受 Git 管理的 `institutions.yaml`，应保留对应仓库版本。

恢复必须使用全新的空数据卷；命令会拒绝覆盖现有数据。在替换服务器上先配置 `deploy/.env`、构建镜像，再于启动 API **之前**恢复：

```sh
./deploy/manage.sh build
./deploy/manage.sh restore /absolute/path/to/paperleet-backup.tar.gz
./deploy/manage.sh up
```

若要在同一服务器演练恢复，同时保留原始数据卷，请全程使用新的 Compose 项目名。先停止旧部署，释放 80 和 443 端口：

```sh
./deploy/manage.sh down
COMPOSE_PROJECT_NAME=paperleet-restore ./deploy/manage.sh restore /absolute/path/to/paperleet-backup.tar.gz
COMPOSE_PROJECT_NAME=paperleet-restore ./deploy/manage.sh up
COMPOSE_PROJECT_NAME=paperleet-restore ./deploy/manage.sh status
```

原项目数据仍然保留。需要返回原部署时，用相同 `COMPOSE_PROJECT_NAME` 前缀停止恢复项目，再不带此前缀执行 `./deploy/manage.sh up`。确认数据不再需要后再删除对应数据卷。

## 邀请用户前的验证

仓库测试覆盖线上来源与邀请限制、账号隐私、练习数据归属、报告完整性、容器限制和部署配置。worker 镜像构建完成后，可以在目标服务器执行：

```sh
PAPERLEET_DOCKER_SMOKE=1 .venv/bin/python -m pytest tests/test_hosted_judge.py -k live_docker -q
PAPERLEET_TEST_DEPLOYMENT=1 .venv/bin/python -m pytest tests/test_deployment_live.py -q
```

这两个可选开发检查需要仓库的 Python 测试环境；正式容器不要求宿主机安装 Python。第一条测试全部 60 个参考实现、未完成 starter 的失败结果、隔离探针和超时清理。第二条要求已经构建全部四个部署镜像；它用临时 Compose 项目、模拟凭据和本机 HTTPS 测试私人练习、真实判题、重启持久性、备份与空数据卷恢复，随后清理测试资源。它不会发送邮件，也不会发布公网网站。实际部署还需检查 DNS、公网可信 TLS、真实邮件送达、备份存储与恢复流程。
