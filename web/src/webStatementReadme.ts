export type StatementExecution='browser'|'server';

/** Statement-only presentation stays with the deferred reader. Source Markdown is immutable. */
export function webStatementReadme(content:string,execution:StatementExecution,t:(zh:string,en:string)=>string){
  return content.replace(
    /##\s*6[.、．]?\s*(?:怎么运行|编写与运行|在网页中运行|Write and run|Run your implementation|Running)[^\n]*\n[\s\S]*?(?=\n##\s*7|$)/,
    `## 6. ${t("编写与运行", "Write and run")}\n\n${execution==='browser' ? t("在右侧完成实现，点击「运行测试」，通过后解锁坑点。注册或登录后提交，可保存解答和成绩到「我的提交」。", "Write code on the right, then Run tests to unlock Notes. Join or sign in, then Submit to save your solution and results in My submissions.") : t("在右侧完成实现，登录后 Submit。服务端会判题并保存解答和结果，可在「我的提交」中查看。", "Write code on the right, then sign in and Submit. The server judges your solution and saves it in My submissions.")}\n\n${t("游客草稿自动保存在当前浏览器，刷新后恢复，不会自动并入账号。", "Guest drafts save in this browser and survive reloads, separately from your account.")}\n`,
  );
}
