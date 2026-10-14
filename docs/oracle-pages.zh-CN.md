# GitHub Pages 与 Oracle 部署

公开站点：`https://paperleet.github.io`。账号应用：`https://app.paperleet.com`。
Pages 构建设置 `VITE_PUBLIC_SITE=1` 和 `VITE_HOSTED_APP_URL=https://app.paperleet.com`；该构建不请求账号、判题或统计 API。登录、社区、主页和投稿入口直接进入账号应用。浏览器练习与草稿保存在 Pages 的浏览器存储中，不会自动迁移到另一域名；切换服务端判题时先下载代码，再到账号应用粘贴继续。账号应用使用普通构建和原有同源 Cookie 登录。

Oracle 的 80/443 由已有 Nginx 管理。安装 Compose 后，设置 `deploy/.env`，使用 `./deploy/oracle.sh build`、`up`、`status`、`backup`、`restore`。所有维护操作使用同一个 wrapper，Compose 项目名为 `paperleet-oracle`。代理只绑定 `127.0.0.1:18880`，API 和判题管理器均不发布主机端口。

在 DNS 添加 `app.paperleet.com` 到 Oracle 的 A 记录。签发证书后添加 `deploy/nginx.oracle.conf` 作为独立 Nginx 配置，先 `nginx -t` 再 reload。Nginx 必须覆盖 `X-Real-IP`；不要将此 Caddy 配置放在可公开直接访问的端口。暂不要变更现有根域名，以保留旧站点直到数据迁移完成。

旧部署数据需由合作方在电脑的部署目录执行 `./deploy/tunnel.sh backup`（隧道部署）或 `./deploy/manage.sh backup`（直接部署）。备份含私人账号与练习数据，应私下传输。另行保存部署配置及认证服务配置，不能提交 Git。Oracle 恢复使用全新的空数据卷；服务运行后创建了数据的卷不能直接覆盖。先恢复到独立 Compose 项目、验证账号与数据，再切换入口。开发数据库不能替代旧站点正式数据。详细流程见 hosting.zh-CN.md。
