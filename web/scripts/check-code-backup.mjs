import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';

export async function checkCodeBackup(server) {
  const { createCodeBackup, serializeCodeBackup, parseCodeBackup, sourceBytes, MAX_BACKUP_BYTES, MAX_RESTORE_BYTES, MAX_BACKUP_SOURCES } = await server.ssrLoadModule('/src/codeBackup.ts');
  const { BackupSourceReview, default: WorkspaceBackup } = await server.ssrLoadModule('/src/WorkspaceBackup.tsx');
  const { WorkspaceBackupFailure } = await server.ssrLoadModule('/src/WorkspaceBackupFallback.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { draftSnapshotMatches } = await server.ssrLoadModule('/src/practiceRecords.ts');
  const source = '# <script>🧪</script>\r\n变量 = "你好"\r\n';
  const accepted = { tests: [{ passed: true }] };
  const saved = { 'module/a': { code: source, acceptedReport: accepted, report: accepted }, 'module/empty': { code: '' }, 'module/report-only': { report: accepted } };
  const backups = [{ owner: 'local', id: 'module/a', code: 'older' }, { owner: 'local', id: 'module/a', code: source }, { owner: 'anonymous', id: 'module/private', code: 'other owner' }];
  const backup = createCodeBackup('local', saved, backups, { 'module/a': { code: 'other tab' } });
  assert.deepEqual(backup.sources, [{ id: 'module/a', code: source, kind: 'draft' }, { id: 'module/empty', code: '', kind: 'draft' }, { id: 'module/a', code: 'other tab', kind: 'conflict' }, { id: 'module/a', code: 'older', kind: 'recovered' }]);
  const text = serializeCodeBackup(backup);
  assert(!text.includes('acceptedReport') && !text.includes('other owner') && !text.includes('owner') && !text.includes('report-only'));
  assert.deepEqual(parseCodeBackup(text), backup, 'Exact Unicode, CRLF and empty source survive export/import');
  assert.deepEqual(parseCodeBackup('\ufeff' + text), backup);
  assert.equal(parseCodeBackup(serializeCodeBackup(createCodeBackup('anonymous', {}, [], {}))).sources.length, 0);
  const injected = JSON.parse(text); injected.owner = 'user:admin'; injected.reports = [accepted]; injected.sources[0].acceptedReport = accepted;
  assert.deepEqual(parseCodeBackup(JSON.stringify(injected)), backup, 'Untrusted grades and account fields cannot become imported workspace data');
  const badRows = [null, {}, { id: '__proto__', code: source, kind: 'draft' }, { id: '../secrets', code: source, kind: 'draft' }, { id: 'module/a', code: 42, kind: 'draft' }, { id: 'module/a', code: source, kind: 'accepted' }];
  for (const row of badRows) assert.throws(() => parseCodeBackup(JSON.stringify({ ...backup, sources: [row] })));
  for (const value of ['{', 'null', '[]', JSON.stringify({ ...backup, version: 2 }), JSON.stringify({ ...backup, format: 'foreign' }), JSON.stringify({ ...backup, sources: {} })]) assert.throws(() => parseCodeBackup(value));
  assert.throws(() => parseCodeBackup('x'.repeat(MAX_BACKUP_BYTES + 1)), /too_large/);
  assert.throws(() => parseCodeBackup(JSON.stringify({ ...backup, sources: [{ id: 'module/a', code: '🧪'.repeat(MAX_BACKUP_BYTES / 3), kind: 'draft' }] })), /too_large/);
  assert.throws(() => parseCodeBackup(JSON.stringify({ ...backup, sources: Array(MAX_BACKUP_SOURCES + 1).fill(backup.sources[0]) })), /too_many/);
  assert.throws(() => serializeCodeBackup({ ...backup, sources: [{ ...backup.sources[0], code: 'x'.repeat(MAX_BACKUP_BYTES) }] }), /too_large/);
  assert.equal(sourceBytes('🧪'), 4);
  const storage = globalThis.localStorage;
  try {
    for (const en of [false, true]) {
      globalThis.localStorage = { getItem: () => en ? 'en' : 'zh' };
      const props = { row: backup.sources[0], problem: { id: 'module/a', number: 1, title: 'Test <tag>', starter: 'starter' }, saved: { code: 'current', updated: 'v1' }, expected: { owner: 'local', code: 'current', updated: 'v1' }, owner: 'local', blocked: '', onReview() {}, onImport() {}, onDownload() {} };
      const render = changes => renderToString(React.createElement(LocaleProvider, null, React.createElement(BackupSourceReview, { ...props, ...changes })));
      const html = render({});
      assert(html.includes(en ? 'Back up current code and load' : '备份当前代码并载入'));
      assert(html.includes('&lt;script&gt;') && !html.includes('<script>') && !html.includes('disabled=""'));
      for (const changes of [{ saved: { code: 'newer', updated: 'v2' } }, { owner: 'user:bob' }, { blocked: 'saving' }, { row: { ...props.row, code: 'current' } }, { row: { ...props.row, code: 'x'.repeat(MAX_RESTORE_BYTES + 1) } }]) assert(render(changes).includes('disabled=""'));
      assert(!render({ problem: undefined }).includes(en ? 'Back up current code and load' : '备份当前代码并载入'), 'Retired modules can be downloaded but cannot be written to arbitrary IDs');
      assert(render({ row: backup.sources[1] }).includes(en ? 'Empty file (0 bytes)' : '空文件（0 字节）'));
      const partial = renderToString(React.createElement(LocaleProvider, null, React.createElement(WorkspaceBackup, {
        owner:'user:alice',problems:[],saved,backups:[],conflicts:{},ready:false,blocked:'offline',onClose(){},onReplace(){return false;},onDownload(){},recovery:{blocked:true,reason:'offline',reload(){}},
      })));
      assert(partial.includes(en ? 'not fully loaded' : '尚未完整加载') && !partial.includes('disabled=""'), 'A resolved owner can export cached code while workspace loading fails; the partial scope must be explicit');
      for(const blocked of [false,true]) {
        const fallback=renderToString(React.createElement(LocaleProvider,null,React.createElement(WorkspaceBackupFailure,{onClose(){},recovery:{blocked,reason:blocked?'saving draft':'',reload(){}}})));
        assert(fallback.includes(en?'download code from each module':'各题编辑器的下载按钮'));
        assert.equal(fallback.includes('disabled=""'),blocked);
        assert(!fallback.includes(en?'Reload page':'刷新页面'),'A failed backup chunk must not offer an unguarded page reload');
      }
    }
  } finally { if (storage === undefined) delete globalThis.localStorage; else globalThis.localStorage = storage; }
  assert(draftSnapshotMatches({ code: '', updated: 'v1' }, { owner: 'local', code: '', updated: 'v1' }));
  assert(!draftSnapshotMatches({ code: '', updated: 'v2' }, { owner: 'local', code: '', updated: 'v1' }));
  console.log('PASS portable code backups: exact multi-module/empty/Unicode sources, recoverable/conflict versions, owner filtering, grade-free allowlist, file limits, stale review gates and bilingual safe previews');
}
