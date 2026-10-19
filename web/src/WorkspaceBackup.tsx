import {appLocation} from './appLocation';
import { useEffect, useId, useRef, useState } from 'react';
import { useI18n } from './i18n';
import { draftSnapshotMatches, type DraftExpectation } from './practiceRecords';
import type { BrowserBackup } from './browserWorkspaceStore';
import type { Problem, Saved } from './types';
import type { EditorRecovery } from './EditorFallback';
import { createCodeBackup, MAX_BACKUP_BYTES, MAX_RESTORE_BYTES, parseCodeBackup, serializeCodeBackup, sourceBytes, type BackupSource, type CodeBackup } from './codeBackup';
import './workspace-backup.css';

export function BackupSourceReview({ row, problem, saved, expected, owner, blocked, onReview, onImport, onDownload }: {
  row: BackupSource; problem?: Problem; saved?: Saved; expected: DraftExpectation; owner: string; blocked: string;
  onReview: () => void; onImport: () => void; onDownload: () => void;
}) {
  const { t } = useI18n();
  const current = saved?.code ?? problem?.starter;
  const matches = expected.owner === owner && draftSnapshotMatches(saved, expected);
  const oversized = sourceBytes(row.code) > MAX_RESTORE_BYTES;
  const identical = current === row.code;
  return <section className="backup-source-review" aria-label={t('核对待载入代码', 'Review code to load')}>
    <h3>{problem ? `${problem.number}. ${problem.title}` : row.id}</h3>
    <p>{t('备份中的代码', 'Source from backup')}</p>
    <pre tabIndex={0}>{row.code || t('空文件（0 字节）', 'Empty file (0 bytes)')}</pre>
    {problem && <details><summary>{t('查看当前代码', 'Review current source')}</summary>
      <pre tabIndex={0}>{current || t('空文件（0 字节）', 'Empty file (0 bytes)')}</pre>
    </details>}
    {!problem && <p role="status">{t('这道题暂不在当前题库中。可以下载代码保留，待题目可用后再载入。', 'This module is not in the current catalog. Download its source and keep it until the module becomes available.')}</p>}
    {problem && !matches && <p role="alert">{t('当前代码或工作区已变化，请重新核对。', 'The current source or workspace changed. Review it again.')}</p>}
    {identical && <p role="status">{t('这份代码与当前代码相同，无需重复载入。', 'This source already matches the current code.')}</p>}
    {oversized && <p role="status">{t('代码超过单题 128 KiB 的载入限制，可以下载后整理。', 'This source exceeds the 128 KiB per-module loading limit. Download it to edit locally.')}</p>}
    {blocked && <p role="status">{blocked}</p>}
    <div className="backup-actions">
      <button className="button subtle" type="button" onClick={onDownload}>{t('下载这份代码', 'Download this source')}</button>
      {problem && !matches && <button className="button subtle" type="button" onClick={onReview}>{t('核对最新代码', 'Review latest source')}</button>}
      {problem && <button className="button" type="button" disabled={!!blocked || !matches || identical || oversized} onClick={onImport}>{t('备份当前代码并载入', 'Back up current code and load')}</button>}
    </div>
  </section>;
}

export default function WorkspaceBackup({ owner, problems, saved, backups, conflicts, ready, blocked, onClose, onReplace, onDownload }: {
  owner: string; problems: Problem[]; saved: Record<string, Saved>; backups: BrowserBackup[]; conflicts: Record<string, Saved>;
  recovery: EditorRecovery;
  ready: boolean; blocked: string; onClose: () => void;
  onReplace: (problem: Problem, source: string, expected: DraftExpectation) => boolean;
  onDownload: (name: string, source: string) => void;
}) {
  const { t } = useI18n();
  const dialog = useRef<HTMLDialogElement>(null), closeButton = useRef<HTMLButtonElement>(null);
  const titleId = useId(), fileId = useId(), sourceId = useId(), readVersion = useRef(0);
  const [backup, setBackup] = useState<CodeBackup>();
  const [review, setReview] = useState<{ index: number; expected: DraftExpectation }>();
  const [reading, setReading] = useState(false), [error, setError] = useState('');
  useEffect(() => {
    const node = dialog.current;
    node?.showModal(); closeButton.current?.focus();
    return () => { ++readVersion.current; if (node?.open) node.close(); };
  }, []);
  function close() { dialog.current?.close(); onClose(); }
  function formatError(cause: unknown) {
    return cause instanceof Error && cause.message === 'too_large'
      ? t('备份超过 10 MiB，无法处理。请保留原文件，并按题下载代码。', 'This backup exceeds 10 MiB. Keep the original file and download code by module.')
      : cause instanceof Error && cause.message === 'too_many'
        ? t('备份超过 1024 份代码的限制。请保留原文件。', 'This backup exceeds the 1024-source limit. Keep the original file.')
        : t('无法读取 PaperLeet 代码备份。请选择由「下载代码备份」导出的 JSON 文件。', 'Could not read this PaperLeet backup. Choose a JSON file exported by Download code backup.');
  }
  function capture(index: number, from = backup) {
    const row = from?.sources[index];
    if (!row) { setReview(undefined); return; }
    const previous = saved[row.id];
    setReview({ index, expected: { owner, code: previous?.code, updated: previous?.updated } });
    setError('');
  }
  async function read(file: File) {
    const version = ++readVersion.current;
    setReading(true); setError(''); setBackup(undefined); setReview(undefined);
    try {
      if (file.size > MAX_BACKUP_BYTES) throw new Error('too_large');
      const parsed = parseCodeBackup(await file.text());
      if (version !== readVersion.current) return;
      setBackup(parsed); capture(0, parsed);
    } catch (cause) { if (version === readVersion.current) setError(formatError(cause)); }
    finally { if (version === readVersion.current) setReading(false); }
  }
  function exportCode() {
    if (owner === 'unresolved') return;
    try {
      onDownload('paperleet-code-backup.json', serializeCodeBackup(createCodeBackup(owner, saved, backups, conflicts)));
      setError('');
    } catch (cause) { setError(formatError(cause)); }
  }
  const row = review && backup?.sources[review.index];
  const problem = row && problems.find(entry => entry.id === row.id);
  function importCode() {
    if (!row || !problem || !review || blocked || !ready || reading || sourceBytes(row.code) > MAX_RESTORE_BYTES
      || review.expected.owner !== owner || !draftSnapshotMatches(saved[row.id], review.expected)) return;
    try {
      onDownload(`paperleet-${row.id.replaceAll('/', '-')}-before-import.py`, saved[row.id]?.code ?? problem.starter);
      if (!onReplace(problem, row.code, review.expected)) {
        setError(t('未能确认代码已载入。请检查当前代码和保存状态，再核对最新代码。', 'The code could not be confirmed as loaded. Check the current source and save status, then review again.'));
        return;
      }
      close();
      appLocation.hash = `/${problem.number}`;
    } catch {
      setError(t('未能完成载入，请检查备份下载和工作区保存状态。', 'Loading could not finish. Check the backup download and workspace save status.'));
    }
  }
  const kinds = { draft: t('草稿', 'Draft'), conflict: t('冲突版本', 'Conflict'), recovered: t('可恢复版本', 'Recoverable version') };
  return <dialog ref={dialog} className="workspace-backup-dialog" aria-labelledby={titleId} onCancel={event => { event.preventDefault(); close(); }}>
    <div className="backup-dialog-heading"><h2 id={titleId}>{t('代码备份与恢复', 'Code backup and restore')}</h2>
      <button ref={closeButton} className="button subtle" type="button" onClick={close}>{t('关闭', 'Close')}</button>
    </div>
    <p>{owner.startsWith('user:') ? t('当前账号工作区', 'Current account workspace') : t('当前练习工作区', 'Current practice workspace')}</p>
    <p>{t('导出当前已加载的草稿、冲突版本与可恢复版本。备份包含完整代码，请妥善保存。', 'Export loaded drafts, conflicting source and recoverable versions. The backup contains complete code; keep it somewhere safe.')}</p>
    {!ready && <p role="status">{t('工作区尚未完整加载，此时备份只包含已加载的代码；完整加载后可再次导出。', 'The workspace is not fully loaded. This backup includes only loaded source; export again after loading finishes.')}</p>}
    <button className="button subtle" type="button" disabled={owner === 'unresolved'} onClick={exportCode}>{t('下载代码备份', 'Download code backup')}</button>
    <hr />
    <h3>{t('从备份载入', 'Load from backup')}</h3>
    <p>{t('选择备份文件后逐份核对代码。载入前会下载当前代码备份；此前通过记录保留，载入的代码需重新运行测试。', 'Choose a backup and review each source before loading. Loading starts a backup download of the current code. Earlier accepted results stay; run the loaded code again to test it.')}</p>
    <div className="backup-file-label"><label htmlFor={fileId}>{t('选择 PaperLeet 代码备份', 'Choose a PaperLeet code backup')}</label>
      <input id={fileId} type="file" accept=".json,application/json" onChange={event => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; if (file) void read(file); }} />
    </div>
    {reading && <p role="status">{t('正在读取备份…', 'Reading backup…')}</p>}
    {error && <p role="alert">{error}</p>}
    {backup && <>
      <p role="status">{t(`备份中有 ${backup.sources.length} 份代码。`, `${backup.sources.length} sources in this backup.`)}</p>
      {!!backup.sources.length && <div className="backup-source-label"><label htmlFor={sourceId}>{t('选择备份中的代码', 'Choose source from backup')}</label>
        <select id={sourceId} value={review?.index ?? 0} onChange={event => capture(Number(event.target.value))}>
          {backup.sources.map((source, index) => {
            const entry = problems.find(p => p.id === source.id);
            return <option key={index} value={index}>{index + 1}. {entry ? entry.title : source.id} · {kinds[source.kind]}</option>;
          })}
        </select>
      </div>}
    </>}
    {row && review && <BackupSourceReview row={row} problem={problem || undefined} saved={saved[row.id]} expected={review.expected} owner={owner}
      blocked={blocked || (!ready ? t('工作区尚未加载完成。', 'The workspace has not finished loading.') : '')}
      onReview={() => capture(review.index)} onImport={importCode}
      onDownload={() => onDownload(`paperleet-${row.id.replaceAll('/', '-')}.py`, row.code)} />}
  </dialog>;
}
