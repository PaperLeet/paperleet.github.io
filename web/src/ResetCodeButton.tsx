import { useEffect, useId, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { useI18n } from './i18n';
import { draftSnapshotMatches, type DraftExpectation } from './practiceRecords';
import type { Problem, Saved } from './types';

type Review = { expected: DraftExpectation; source: string; starter: string; problem: string };
export function ResetCodeButton({ owner, problem, saved, ready, busy, conflict, failed, signedIn, onReset }: {
  owner: string; problem: Problem; saved: Saved; ready: boolean; busy: boolean; conflict: boolean; failed: boolean; signedIn: boolean;
  onReset: (expected: DraftExpectation) => boolean;
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
      if (onReset(review.expected)) setReview(undefined);
      else setError(true);
    } catch { setError(true); }
  }
  return <>
    <button className="icon-button" type="button" disabled={unavailable || source === problem.starter}
      title={t('核对并恢复初始代码', 'Review and restore starter')}
      aria-label={t('恢复初始代码', 'Restore starter code')} onClick={capture}>
      <RotateCcw size={14} />
    </button>
    <dialog ref={dialog} className="reset-code-dialog" aria-labelledby={titleId} onCancel={() => setReview(undefined)}>
      {review && <>
        <h2 id={titleId}>{t('恢复本题初始代码？', 'Restore this module’s starter?')}</h2>
        <p>{t('确认后将替换编辑器中的代码。已提交的解答和成绩会保留。',
          'This replaces the editor code. Submitted solutions and results are kept.')}</p>
        <p>{signedIn ? t('账号草稿会同步更新。之前提交的代码可在「我的提交」中查看。',
          'Your account draft will sync. Previous submitted code is available in My submissions.')
          : t('这会替换当前浏览器中的草稿。如需保留当前代码，请先复制或下载。',
          'This replaces the draft in this browser. Copy or download your current code first if you want to keep it.')}</p>
        <details><summary>{t('查看将被替换的代码', 'Review code to be replaced')}</summary>
          <pre tabIndex={0}>{review.source || t('空文件（0 字节）', 'Empty file (0 bytes)')}</pre>
        </details>
        {!matches && <p role="alert">{t('代码或工作区已变化，请重新核对后再恢复。', 'The code or workspace changed. Review the latest source before restoring.')}</p>}
        {unavailable && <p role="status">{t('请先等待保存完成，或解决工作区的保存错误和草稿冲突。', 'Wait for saving to finish, or resolve workspace save errors and draft conflicts first.')}</p>}
        {error && <p role="alert">{t('未能完成恢复。请检查当前草稿后重试。', 'Restore could not finish. Check the current draft and retry.')}</p>}
        <div className="reset-code-actions">
          <button ref={cancel} className="button subtle" type="button" onClick={() => setReview(undefined)}>{t('保留当前代码', 'Keep current code')}</button>
          {!matches && <button className="button subtle" type="button" disabled={unavailable} onClick={capture}>{t('核对最新代码', 'Review latest source')}</button>}
          <button className="button" type="button" disabled={unavailable || !matches} onClick={reset}>{t('恢复初始代码', 'Restore starter')}</button>
        </div>
      </>}
    </dialog>
  </>;
}
