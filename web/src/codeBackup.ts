import type { BrowserBackup } from './browserWorkspaceStore';
import type { Saved } from './types';

export const MAX_BACKUP_BYTES = 10 * 1024 * 1024;
export const MAX_BACKUP_SOURCES = 1024;
export const MAX_RESTORE_BYTES = 128 * 1024;
export type BackupSource = { id: string; code: string; kind: 'draft' | 'recovered' | 'conflict' };
export type CodeBackup = { format: 'paperleet-code-backup'; version: 1; sources: BackupSource[] };
export class BackupFormatError extends Error {}
const validId = (id: unknown): id is string => typeof id === 'string' && id.length <= 160 && /^[a-z0-9][a-z0-9_-]*(?:\/[a-z0-9][a-z0-9_-]*)+$/.test(id);
export const sourceBytes = (code: string) => new TextEncoder().encode(code).byteLength;

/** Export visible code only; no account identity, reports or acceptance authority. */
export function createCodeBackup(owner: string, saved: Record<string, Saved>, backups: BrowserBackup[], conflicts: Record<string, Saved>): CodeBackup {
  const sources: BackupSource[] = [], seen = new Map<string, Set<string>>();
  function add(id: string, code: string | undefined, kind: BackupSource['kind']) {
    if (code === undefined) return;
    if (!validId(id)) throw new BackupFormatError('invalid_id');
    const codes = seen.get(id) || new Set<string>();
    if (codes.has(code)) return;
    codes.add(code); seen.set(id, codes); sources.push({ id, code, kind });
  }
  for (const [id, entry] of Object.entries(saved)) add(id, entry.code, 'draft');
  for (const [id, entry] of Object.entries(conflicts)) add(id, entry.code, 'conflict');
  for (const entry of backups) if (entry.owner === owner) add(entry.id, entry.code, 'recovered');
  if (sources.length > MAX_BACKUP_SOURCES) throw new BackupFormatError('too_many');
  return { format: 'paperleet-code-backup', version: 1, sources };
}

export function serializeCodeBackup(backup: CodeBackup): string {
  const text = JSON.stringify(backup, null, 2) + '\n';
  if (sourceBytes(text) > MAX_BACKUP_BYTES) throw new BackupFormatError('too_large');
  return text;
}

/** Reconstruct an allowlisted shape. Extra report/identity fields are never restored. */
export function parseCodeBackup(text: string): CodeBackup {
  if (text.length > MAX_BACKUP_BYTES || sourceBytes(text) > MAX_BACKUP_BYTES) throw new BackupFormatError('too_large');
  let value;
  try { value = JSON.parse(text.charCodeAt(0) === 0xfeff ? text.slice(1) : text); }
  catch { throw new BackupFormatError('invalid'); }
  if (!value || value.format !== 'paperleet-code-backup' || value.version !== 1 || !Array.isArray(value.sources)) throw new BackupFormatError('invalid');
  if (value.sources.length > MAX_BACKUP_SOURCES) throw new BackupFormatError('too_many');
  const sources: BackupSource[] = [];
  for (const row of value.sources) {
    if (!row || !validId(row.id) || typeof row.code !== 'string' || !['draft', 'recovered', 'conflict'].includes(row.kind)) throw new BackupFormatError('invalid');
    sources.push({ id: row.id, code: row.code, kind: row.kind });
  }
  return { format: 'paperleet-code-backup', version: 1, sources };
}
