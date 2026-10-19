/** Immediate submission feedback, independent of the deferred job transport. */
export const submissionModuleLoadError = "The submission component could not load. This submission was not sent. Reload the page, then submit again.";
export const pendingSubmissionModuleLoadError = "The submission component could not load. Reload the page to reconnect to your existing submission.";
export function jobErrorText(message: string, locale: string): string {
  if (locale !== "zh") return message;
  const translations: Record<string, string> = {
    [submissionModuleLoadError]: "提交组件加载失败，本次尚未发送。刷新页面后，请再次提交。",
    [pendingSubmissionModuleLoadError]: "提交组件加载失败。请刷新页面，恢复查询已有任务的结果。",
    "The judge queue is full. Your draft is preserved; try again shortly.": "判题队列已满，请稍后再提交；你的草稿已保留。",
    "You already have a queued or running submission. Wait for it to finish.": "你已有一个排队或运行中的任务，请等待该任务完成。",
    "This job has already started and cannot be removed from the queue.": "任务已经开始执行，无法取消排队，请等待结果。",
    "This job does not exist or has expired. Your draft is preserved.": "任务不存在或已过期；草稿仍保留，可以重新提交。",
    "The service restarted during this run. Your draft is preserved; submit again.": "服务在判题时重启，本次任务已中断；草稿保留，请重新提交。",
    "The queue wait limit was reached. Your draft is preserved; submit again.": "排队等待已超时；草稿保留，请重新提交。",
    "The judge could not complete this job. Your draft is preserved.": "本次判题未能完成；草稿保留，请稍后重试。",
    "Sign in to view or submit judge jobs.": "请登录后查看或提交判题任务。",
    "Your account changed. Reload before submitting.": "账号已变更，请刷新页面后再提交。",
    "The judge job service is temporarily unavailable. Your draft is preserved.": "判题任务服务暂时不可用；你的草稿已保留。",
    "The queue stayed full. Automatic waiting stopped; your draft is preserved.": "队列持续繁忙，已停止自动等待；草稿仍保留，可稍后重新提交。",
    "The submission is being confirmed. Wait for its queue status before cancelling.": "正在确认这次提交，确认入队后才能取消排队。",
    "Server submission could not start. Your draft is preserved; try again.": "服务端提交未能启动；草稿已保留，请重试。",
    "Too many requests. Please wait before submitting again.": "提交过于频繁，请稍候再试。",
  };
  return translations[message] || message;
}
