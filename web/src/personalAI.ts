import { useEffect, useRef, useState } from 'react';
import { socialApi } from './social';

export type Provider = 'gemini' | 'openai' | 'anthropic' | 'kimi' | 'qwen' | 'deepseek';
export type Region = 'international' | 'china';
export type ProviderState = { id: Provider; configured: boolean; model: string | null; updated_at: string | null; region: Region | null };
export type Settings = { user_id: string; available: boolean; active_provider: Provider | null; revision: number; providers: ProviderState[] };
export const providers: { id: Provider; label: string }[] = [{ id: 'gemini', label: 'Gemini' }, { id: 'openai', label: 'OpenAI' }, { id: 'anthropic', label: 'Claude' }, { id: 'kimi', label: 'Kimi' }, { id: 'qwen', label: 'Qwen' }, { id: 'deepseek', label: 'DeepSeek' }];
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
export const providerId = (value: unknown): value is Provider => providers.some(item => item.id === value);
export const regionalProvider = (value: Provider) => value === 'kimi' || value === 'qwen';
export const providerLabel = (value?: string | null) => providers.find(item => item.id === value)?.label || value || '';
export const modelId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$/.test(value);
export function readSettings(value: unknown, owner: string): Settings {
  if (!record(value) || value.user_id !== owner || typeof value.available !== 'boolean' || !Number.isSafeInteger(value.revision) || Number(value.revision) < 0 || value.active_provider !== null && !providerId(value.active_provider) || !Array.isArray(value.providers)) throw Error('invalid_response');
  const rows = value.providers.map<ProviderState>(item => {
    if (!record(item) || !providerId(item.id) || typeof item.configured !== 'boolean' || item.model !== null && !modelId(item.model) || item.updated_at !== null && (typeof item.updated_at !== 'string' || !Number.isFinite(Date.parse(item.updated_at)))) throw Error('invalid_response');
    const region = item.region ?? (regionalProvider(item.id) ? 'international' : null);
    if (regionalProvider(item.id) ? region !== 'international' && region !== 'china' : region !== null) throw Error('invalid_response');
    return { id: item.id, configured: item.configured, model: item.model as string | null, updated_at: item.updated_at as string | null, region: region as Region | null };
  });
  // The original three remain usable while the server rolls out additional providers.
  if (rows.length < 3 || rows.length > providers.length || new Set(rows.map(item => item.id)).size !== rows.length || !['gemini', 'openai', 'anthropic'].every(id => rows.some(item => item.id === id)) || value.active_provider !== null && !rows.some(item => item.id === value.active_provider && item.configured && item.model)) throw Error('invalid_response');
  return { user_id: owner, available: value.available, active_provider: value.active_provider as Provider | null, revision: Number(value.revision), providers: rows };
}

export type PersonalAIStatus = { loading: boolean; configured: boolean; provider?: string | null; model: string | null; available?: boolean; error?: boolean };
export function readPersonalAIStatus(value: unknown, owner: string): PersonalAIStatus {
  if (!record(value) || value.user_id !== owner || typeof value.configured !== 'boolean' || typeof value.available !== 'boolean' || value.configured && (value.source !== 'personal' || !value.available || !providerId(value.provider) || !modelId(value.model))) throw Error('invalid_response');
  return { loading: false, configured: value.configured, available: value.available, provider: value.configured ? value.provider as Provider : null, model: value.configured ? value.model as string : null };
}
export function usePersonalAIStatus(owner: string | undefined, supported: boolean) {
  const [revision, setRevision] = useState(0);
  const [value, setValue] = useState<PersonalAIStatus & { owner?: string }>({ loading: true, configured: false, model: null });
  const current = useRef({ owner, supported }); current.current = { owner, supported };
  const reading = useRef(0), checks = useRef(new Set<AbortController>());
  useEffect(() => {
    const controller = new AbortController(), sequence = ++reading.current; let active = true;
    setValue({ owner, loading: !!owner && supported, configured: false, model: null });
    if (!owner || !supported) return () => controller.abort();
    const timeout = setTimeout(() => controller.abort(), 20000);
    void socialApi<unknown>('account/ai-settings/status', undefined, controller.signal).then(result => {
      const status = readPersonalAIStatus(result, owner);
      if (!active || sequence !== reading.current || current.current.owner !== owner || !current.current.supported) return;
      setValue({ owner, ...status });
    }).catch(() => { if (active && sequence === reading.current && current.current.owner === owner) setValue({ owner, loading: false, configured: false, model: null, error: true }); }).finally(() => clearTimeout(timeout));
    return () => { active = false; reading.current++; controller.abort(); checks.current.forEach(item => item.abort()); checks.current.clear(); clearTimeout(timeout); };
  }, [owner, supported, revision]);
  const state: PersonalAIStatus = value.owner === owner && supported ? value : { loading: !!owner && supported, configured: false, model: null };
  function refresh() { setValue({ owner, loading: !!owner && supported, configured: false, model: null }); setRevision(value => value + 1); }
  async function verify() {
    if (!owner || !supported || current.current.owner !== owner || !current.current.supported) return false;
    const sequence = ++reading.current, controller = new AbortController(); checks.current.add(controller);
    const timeout = setTimeout(() => controller.abort(), 20000);
    setValue({ owner, loading: true, configured: false, model: null });
    try {
      const status = readPersonalAIStatus(await socialApi<unknown>('account/ai-settings/status', undefined, controller.signal), owner);
      if (sequence !== reading.current || current.current.owner !== owner || !current.current.supported) return false;
      setValue({ owner, ...status }); return status.configured;
    } catch { if (sequence === reading.current && current.current.owner === owner) setValue({ owner, loading: false, configured: false, model: null, error: true }); return false; }
    finally { clearTimeout(timeout); checks.current.delete(controller); }
  }
  return { state, refresh, verify };
}
