type QuotaResponse = { code?: unknown; quota?: { action?: unknown; limit?: unknown; reset_at?: unknown } };

/** Use only validated server fields; the account quota is enforced by the API. */
export function publicationQuotaMessage(data: QuotaResponse, english: boolean): string | undefined {
  const quota = data?.quota;
  if (data?.code !== 'daily_quota_exceeded' || !quota ||
    !['paper', 'post', 'reply'].includes(String(quota.action)) ||
    !Number.isSafeInteger(quota.limit) || Number(quota.limit) < 1 ||
    typeof quota.reset_at !== 'string' || !Number.isFinite(Date.parse(quota.reset_at))) return;
  const action = String(quota.action);
  const label = english ? { paper: 'paper proposals', post: 'posts', reply: 'replies' }[action]
    : { paper: '论文投稿', post: '帖子', reply: '回复' }[action];
  const time = new Date(quota.reset_at).toLocaleString(english ? 'en-US' : 'zh-CN', { timeZoneName: 'short' });
  return english ? `Daily limit reached (${quota.limit} ${label}). You can publish again at ${time}. Your draft is kept.`
    : `今日${label}额度已用完（每日上限 ${quota.limit}）。可在 ${time} 后继续发布，当前草稿已保留。`;
}

export function currentQuotaMessage(data: QuotaResponse) {
  let english = false;
  try { english = localStorage.getItem('paperleet.locale') === 'en'; } catch { /* Default language. */ }
  return publicationQuotaMessage(data, english);
}
