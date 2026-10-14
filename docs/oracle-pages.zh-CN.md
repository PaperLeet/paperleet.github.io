# GitHub Pages 与 Oracle 部署

公开站点：`https://paperleet.github.io`。账号应用的预定域名：`https://app.paperleet.com`，尚未开放公网访问。
Pages 构建设置 `VITE_PUBLIC_SITE=1`、`VITE_HOSTED_APP_AVAILABLE=0` 和 `VITE_HOSTED_APP_URL=https://app.paperleet.com`；该构建不请求账号、判题或统计 API。目前隐藏账号、社区和投稿入口。账号应用就绪后设为 `1`，这些入口将直接进入账号应用。浏览器练习与草稿保存在 Pages 的浏览器存储中，不会自动迁移到另一域名；切换服务端判题时先下载代码，再到账号应用粘贴继续。账号应用使用普通构建和原有同源 Cookie 登录。

Oracle 的 80/443 由已有 Nginx 管理。安装 Compose 后，设置 `deploy/.env`，使用 `./deploy/oracle.sh build`、`up`、`status`、`backup`、`restore`。所有维护操作使用同一个 wrapper，Compose 项目名为 `paperleet-oracle`。代理只绑定 `127.0.0.1:18880`，API 和判题管理器均不发布主机端口。

在 DNS 添加 `app.paperleet.com` 到 Oracle 的 A 记录。签发证书后添加 `deploy/nginx.oracle.conf` 作为独立 Nginx 配置，先 `nginx -t` 再 reload。Nginx 必须覆盖 `X-Real-IP`；不要将此 Caddy 配置放在可公开直接访问的端口。暂不要变更现有根域名，以保留旧站点直到数据迁移完成。

旧部署数据需由合作方在电脑的部署目录执行 `./deploy/tunnel.sh backup`（隧道部署）或 `./deploy/manage.sh backup`（直接部署）。备份含私人账号与练习数据，应私下传输。另行保存部署配置及认证服务配置，不能提交 Git。Oracle 恢复使用全新的空数据卷；服务运行后创建了数据的卷不能直接覆盖。先恢复到独立 Compose 项目、验证账号与数据，再切换入口。开发数据库不能替代旧站点正式数据。详细流程见 hosting.zh-CN.md。

当前只开放 GitHub Pages：设置 `VITE_HOSTED_APP_AVAILABLE=0`，隐藏账号、社区、投稿入口，并禁用服务端判题选择。浏览器题库、判题和本机进度照常工作。Oracle 已部署但尚无公网域名；HTTPS 验证通过后将此开关设为 `1`，再发布 Pages 即可开放入口。旧数据迁移暂不属于本阶段。

学校／公司邮箱注册代码已支持 `PAPERLEET_REGISTRATION_POLICY=work_email`。开放前在 Oracle 配置 Supabase Email OTP 和自定义 SMTP，将 `PAPERLEET_AUTH_MODE` 从 `test` 改为 `institutional`，并验证真实收信、HTTPS、登录及私有草稿。详细规则见 hosting.zh-CN.md。邮件服务与公网入口未就绪时继续保留测试账号模式；已有测试账号不会自动变成邮箱账户。
