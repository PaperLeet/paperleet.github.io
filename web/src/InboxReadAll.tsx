import { CheckCheck } from 'lucide-react';
import { useI18n } from './i18n';
import type { useInboxReadAll } from './inbox';

export default function InboxReadAll({action}: {action: ReturnType<typeof useInboxReadAll>}) {
  const {t} = useI18n();
  return <div className="inbox-read-all">
    <button type="button" className="inbox-read-all-button" disabled={action.disabled} onClick={() => void action.mark()}>
      <CheckCheck size={15}/>{action.busy ? t('正在标记…', 'Marking…') : t('全部标为已读', 'Mark all as read')}
    </button>
    {action.error && <div className="inbox-read-all-feedback error" role="alert"><p>{action.error}</p><button type="button" disabled={action.disabled} onClick={() => void action.mark()}>{t('重试', 'Retry')}</button></div>}
    {action.result && <p className="inbox-read-all-feedback" role="status">{action.result.unread_count
      ? t('先前的消息已标为已读，新消息仍保持未读。', 'Earlier messages marked as read. New messages remain unread.')
      : t('已全部标为已读', 'All marked as read')}</p>}
  </div>;
}
