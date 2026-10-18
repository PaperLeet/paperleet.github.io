import AppLink from './AppLink';
import { useState } from 'react';
import { useI18n } from './i18n';
import type { DraftHandoff } from './draftHandoff';
import type { Problem, Saved } from './types';

export function DraftHandoffNotice({ pending, problem, current, username, open, ready, conflict, onImport, onDiscard }: {
  pending: DraftHandoff; problem: Problem; current: Saved; username: string; open: boolean; ready: boolean; conflict: boolean;
  onImport: (backup: boolean) => boolean; onDiscard: () => void;
}) {
  const { t,locale } = useI18n();
  const [error,setError] = useState(false);
  const title = locale === 'en' ? problem.translations?.en?.title || problem.title : problem.title;
  const different = current.code !== undefined && current.code !== problem.starter && current.code !== pending.code;
  return <section className="draft-handoff-notice" aria-label={t('导入未登录草稿', 'Import your signed-out draft')}>
    <strong>{t(`已为本标签页保留：${title}`, `Held in this tab: ${title}`)}</strong>
    <p>{t(`登录的是 @${username}。导入后仅替换本题代码，成绩需要重新判题。暂存一小时后失效。`, `You are signed in as @${username}. Importing replaces only this module’s code; results require a new run. The held copy expires after one hour.`)}</p>
    {!open ? <AppLink className="button subtle" href={`#/${problem.number}`}>{t('查看本题草稿', 'Review this module’s draft')}</AppLink> : !ready ? <p role="status">{t('正在加载账号草稿…', 'Loading your account draft…')}</p> : <>
      <details><summary>{t('查看待导入代码', 'Review the held code')}</summary><pre>{pending.code}</pre></details>
      {different && <><p>{t('账号已有不同草稿。替换前会下载原草稿备份。', 'Your account has a different draft. A backup downloads before it is replaced.')}</p><details><summary>{t('查看账号现有代码', 'Review the current account code')}</summary><pre>{current.code}</pre></details></>}
      {conflict && <p role="status">{t('请先解决此题的账号草稿冲突，再导入。', 'Resolve this module’s account draft conflict before importing.')}</p>}
      <button className="button" type="button" disabled={conflict} onClick={() => { try { setError(!onImport(different)); } catch { setError(true); } }}>{different ? t('备份并替换草稿', 'Back up and replace draft') : t('导入本题草稿', 'Import this draft')}</button>
    </>}
    <button className="text-button" type="button" onClick={onDiscard}>{t('取消导入', 'Cancel import')}</button>
    {error && <p role="alert">{t('无法导入：账号或草稿可能已变化，或备份下载失败。请检查后重试。', 'Import could not finish. The account or draft may have changed, or the backup download failed. Review and retry.')}</p>}
  </section>;
}
