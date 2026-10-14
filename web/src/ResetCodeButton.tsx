import { useEffect, useId, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { useI18n } from './i18n';
import { draftSnapshotMatches, type DraftExpectation } from './practiceRecords';
import type { Problem, Saved } from './types';

type Review = { expected: DraftExpectation; source: string; starter: string; problem: string };
export function ResetCodeButton({ owner, problem, saved, ready, busy, conflict, failed, browserWorkspace, onReset, onDownload }: {
  owner: string; problem: Problem; saved: Saved; ready: boolean; busy: boolean; conflict: boolean; failed: boolean; browserWorkspace: boolean;
  onReset: (expected: DraftExpectation) => boolean; onDownload: (source: string) => void;
}) {
  const { t } = useI18n();
  const [review, setReview] = useState<Review>();
  const [error, setError] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const source = saved.code ?? problem.starter;
  const unavailable = !ready || busy || conflict || failed;
  const matches = !!review && review.expected.owner === owner && review.problem === problem.id
    && review.starter === problem.starter && draftSnapshotMatches(saved, review.expected);
  useEffect(() => {
    const node = dialog.current;
    if (review && node && !node.open) { node.showModal(); cancel.current?.focus(); }
    else if (!review && node?.open) node.close();
  }, [review]);
  function capture() {
    setError(false);
    setReview({ expected: { owner, code: saved.code, updated: saved.updated }, source, starter: problem.starter, problem: problem.id });
  }
  function reset() {
    if (!review || unavailable || !matches) return;
    try {
      onDownload(review.source);
      if (onReset(review.expected)) setReview(undefined);
      else setError(true);
    } catch { setError(true); }
  }
  return <>
    <button className="icon-button" type="button" disabled={unavailable || source === problem.starter}
      title={t('核对并恢复初始代码', 'Review and restore starter')}
      aria-label={t('恢复初始代码并备份', 'Reset starter with backup')} onClick={capture}>
      <RotateCcw size={14} />
    </button>
    <dialog ref={dialog} className="reset-code-dialog" aria-labelledby={titleId} onCancel={() => setReview(undefined)}>
      {review && <>
        <h2 id={titleId}>{t('恢复本题初始代码？', 'Restore this module’s starter?')}</h2>
        <p>{t('确认后将替换编辑器中的代码，并发起这份代码的备份下载。此前的通过记录会保留。',
          'Confirming replaces the editor code and starts a backup download of the source shown here. Earlier accepted results are kept.')}</p>
        <p>{browserWorkspace ? t('原代码也会随本次保存保留在「可恢复的其他版本」中；请留意工作区的保存状态。',
          'The original source is also kept in Other recoverable versions when this change saves. Check the workspace save status.')
          : t('账号草稿会同步更新。请保留下载的备份，以便需要时恢复。',
          'Your account draft will sync this change. Keep the downloaded backup so you can restore it if needed.')}</p>
        <details><summary>{t('查看待备份代码', 'Review source to back up')}</summary>
          <pre tabIndex={0}>{review.source || t('空文件（0 字节）', 'Empty file (0 bytes)')}</pre>
        </details>
        {!matches && <p role="alert">{t('代码或工作区已变化，请重新核对后再恢复。', 'The code or workspace changed. Review the latest source before restoring.')}</p>}
        {unavailable && <p role="status">{t('请先等待保存完成，或解决工作区的保存错误和草稿冲突。', 'Wait for saving to finish, or resolve workspace save errors and draft conflicts first.')}</p>}
        {error && <p role="alert">{t('未能完成恢复。请检查备份下载和当前草稿，再重试。', 'Restore could not finish. Check the backup download and current draft, then retry.')}</p>}
        <div className="reset-code-actions">
          <button ref={cancel} className="button subtle" type="button" onClick={() => setReview(undefined)}>{t('保留当前代码', 'Keep current code')}</button>
          {!matches && <button className="button subtle" type="button" disabled={unavailable} onClick={capture}>{t('核对最新代码', 'Review latest source')}</button>}
          <button className="button" type="button" disabled={unavailable || !matches} onClick={reset}>{t('备份并恢复初始代码', 'Back up and restore starter')}</button>
        </div>
      </>}
    </dialog>
  </>;
}
