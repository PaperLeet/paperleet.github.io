import { useI18n } from './i18n';

export const DAILY_PUBLICATION_LIMITS = { paper: 1, post: 5, reply: 50 } as const;
export type PublicationAction = keyof typeof DAILY_PUBLICATION_LIMITS;

export function PublishingRules({ action }: { action?: PublicationAction }) {
  const { t } = useI18n();
  const labels = { paper: t('论文投稿', 'paper proposals'), post: t('帖子', 'posts'), reply: t('回复', 'replies') };
  const units = { paper: '篇', post: '个', reply: '条' };
  return <details className="publishing-rules">
    <summary>{action
      ? t(`每日最多 ${DAILY_PUBLICATION_LIMITS[action]} ${units[action]}${labels[action]} · 发布规则`, `Up to ${DAILY_PUBLICATION_LIMITS[action]} ${labels[action]} per day · Publishing rules`)
      : t('社区发布规则', 'Community publishing rules')}</summary>
    <p>{t('每个账号每天最多投稿 1 篇论文、发布 5 个帖子、回复 50 条。UTC 零点重置；删除内容不返还额度。做题代码提交单独计算。',
      'Each account can propose 1 paper, publish 5 posts, and send 50 replies per day. Limits reset at midnight UTC. Deletion does not restore quota. Code submissions have separate limits.')}</p>
    <p>{t('允许 AI 辅助，但发布前须由本人核对内容和引用。禁止无人值守的批量投稿、发帖、自动回复，以及使用多个账号绕过限制。',
      'AI assistance is allowed; review the content and sources yourself before publishing. Unattended bulk proposals, posts, automated replies, and using multiple accounts to bypass limits are prohibited.')}</p>
    <p>{t('论文投稿需经过人工评审后才会进入题库。请勿发布重复内容、广告或他人的私人信息。',
      'Paper proposals require human review before entering the catalog. Do not publish duplicate content, spam, or other people’s private information.')}</p>
  </details>;
}
