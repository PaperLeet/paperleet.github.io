import { useCallback, useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { useI18n } from './i18n';
import { socialApi, useAccount } from './social';
import './ai-settings.css';

type Provider = 'gemini' | 'openai' | 'anthropic';
type ProviderState = { id: Provider; configured: boolean; model: string | null; updated_at: string | null };
type Settings = { user_id: string; available: boolean; active_provider: Provider | null; revision: number; providers: ProviderState[] };
type Model = { id: string; name: string };
const providers: { id: Provider; label: string }[] = [{ id: 'gemini', label: 'Gemini' }, { id: 'openai', label: 'OpenAI' }, { id: 'anthropic', label: 'Claude' }];
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const providerId = (value: unknown): value is Provider => providers.some(item => item.id === value);
const modelId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$/.test(value);
function readSettings(value: unknown, owner: string): Settings {
  if (!record(value) || value.user_id !== owner || typeof value.available !== 'boolean' || !Number.isSafeInteger(value.revision) || Number(value.revision) < 0 || value.active_provider !== null && !providerId(value.active_provider) || !Array.isArray(value.providers)) throw Error('invalid_response');
  const rows = value.providers.map<ProviderState>(item => {
    if (!record(item) || !providerId(item.id) || typeof item.configured !== 'boolean' || item.model !== null && !modelId(item.model) || item.updated_at !== null && (typeof item.updated_at !== 'string' || !Number.isFinite(Date.parse(item.updated_at)))) throw Error('invalid_response');
    return { id: item.id, configured: item.configured, model: item.model as string | null, updated_at: item.updated_at as string | null };
  });
  if (rows.length !== 3 || new Set(rows.map(item => item.id)).size !== 3) throw Error('invalid_response');
  return { user_id: owner, available: value.available, active_provider: value.active_provider as Provider | null, revision: Number(value.revision), providers: rows };
}

export default function AISettings({ onChanged, onClose }: { onChanged?: () => void; onClose?: () => void }) {
  const { user, loading } = useAccount();
  const { t } = useI18n();
  const current = useRef(''); current.current = !loading && user?.id || '';
  const dialog = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!onClose) return;
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLElement>('button')?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.stopPropagation(); onClose(); }
      if (event.key !== 'Tab') return;
      const items = [...(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled)') || [])];
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', keydown);
    return () => { document.removeEventListener('keydown', keydown); if (previous?.isConnected) previous.focus(); };
  }, [onClose]);
  const contents = <section ref={dialog} className={`ai-settings${onClose ? ' ai-settings-dialog' : ' social-panel'}`} role={onClose ? 'dialog' : undefined} aria-modal={onClose ? true : undefined} aria-label={t('AI 设置', 'AI settings')}>
    <header><h3>{t('AI 设置', 'AI settings')}</h3>{onClose && <button type="button" className="ai-settings-close" aria-label={t('关闭 AI 设置', 'Close AI settings')} onClick={onClose}><X size={18} /></button>}</header>
    {!current.current ? <p role="status">{loading ? t('正在确认账号…', 'Checking your account…') : t('登录后配置 AI。', 'Sign in to configure AI.')}</p> : <AISettingsForm key={current.current} owner={current.current} canAct={() => current.current === user?.id && !loading} onChanged={onChanged} />}
  </section>;
  return onClose ? <div className="ai-settings-overlay">{contents}</div> : contents;
}

function AISettingsForm({ owner, canAct, onChanged }: { owner: string; canAct: () => boolean; onChanged?: () => void }) {
  const { t } = useI18n();
  const [settings, setSettings] = useState<Settings>(), [provider, setProvider] = useState<Provider>('gemini');
  const [apiKey, setApiKey] = useState(''), [model, setModel] = useState(''), [models, setModels] = useState<Model[]>([]);
  const [customModel, setCustomModel] = useState(false);
  const [busy, setBusy] = useState('load'), [error, setError] = useState(''), [notice, setNotice] = useState(''), [blocked, setBlocked] = useState(false);
  const generation = useRef(0), active = useRef(true), pending = useRef(false), controllers = useRef(new Set<AbortController>()), owns = useRef(canAct); owns.current = canAct;
  const keyInput = useCallback((node: HTMLInputElement | null) => { if (node) return () => { node.value = ''; }; }, []);
  const current = (epoch: number) => active.current && generation.current === epoch && owns.current();
  const selected = settings?.providers.find(item => item.id === provider);
  const hasKey = !!apiKey.trim() || !!selected?.configured;
  const manualModel = customModel || !!model && !models.some(item => item.id === model);
  const enabled = settings?.available === true && !blocked && !busy;
  function message(reason: unknown) {
    const code = record(reason) && typeof reason.code === 'string' ? reason.code : '', status = record(reason) ? reason.status : undefined;
    if (code === 'account_changed' || status === 401 && !code.includes('provider')) return t('账号已变化，请重新登录。', 'Your account changed. Sign in again.');
    if (status === 409) return t('AI 设置已在其他窗口更改，请重新读取。', 'AI settings changed in another window. Reload settings.');
    if (code === 'ai_settings_unavailable') return t('个人 AI 设置暂不可用。', 'Personal AI settings are unavailable.');
    if (code === 'ai_key_unavailable') return t('已保存的密钥无法读取，请重新填写。', 'The saved key cannot be read. Enter it again.');
    if (code === 'api_key_required') return t('请填写 API key。', 'Enter an API key.');
    if (status === 429) return t('请求过于频繁，请稍后重试。', 'Too many requests. Try again shortly.');
    if (/key|auth|credential/.test(code) || status === 403) return t('密钥未通过验证，请检查权限。', 'The key was rejected. Check its permissions.');
    if (/model/.test(code)) return t('模型不可用，请选择其他模型。', 'This model is unavailable. Choose another model.');
    return t('操作未完成，请重试。', 'Could not complete this action. Try again.');
  }
  async function request(path: string, body?: unknown) {
    const controller = new AbortController(); controllers.current.add(controller);
    const timeout = setTimeout(() => controller.abort(), 60000);
    try { return await socialApi<unknown>(path, body, controller.signal); }
    finally { clearTimeout(timeout); controllers.current.delete(controller); }
  }
  function apply(next: Settings, chosen: Provider) {
    setSettings(next); setProvider(chosen); setModel(next.providers.find(item => item.id === chosen)?.model || '');
    setApiKey(''); setModels([]); setCustomModel(false); setBlocked(false);
  }
  async function load() {
    if (pending.current || !active.current || !owns.current()) return;
    pending.current = true; const epoch = generation.current; setBusy('load'); setError(''); setNotice(''); setApiKey('');
    try { const next = readSettings(await request('account/ai-settings'), owner); if (current(epoch)) apply(next, next.active_provider || 'gemini'); }
    catch (reason) { if (current(epoch)) { setError(message(reason)); setBlocked(true); } }
    finally { if (current(epoch)) { pending.current = false; setBusy(''); } }
  }
  useEffect(() => {
    active.current = true; pending.current = false; void load();
    return () => { active.current = false; generation.current++; controllers.current.forEach(controller => controller.abort()); controllers.current.clear(); };
  }, []);
  function choose(next: Provider) {
    if (pending.current) return;
    setProvider(next); setApiKey(''); setModel(settings?.providers.find(item => item.id === next)?.model || ''); setModels([]); setCustomModel(false); setError(''); setNotice('');
  }
  async function perform(action: 'models' | 'test' | 'save' | 'remove') {
    if (pending.current || !active.current || !owns.current() || !settings || blocked) return;
    if (action !== 'remove' && (!settings.available || !hasKey) || (action === 'test' || action === 'save') && !modelId(model.trim())) return;
    pending.current = true; const epoch = generation.current; setBusy(action); setError(''); setNotice('');
    const body = action === 'remove' ? { provider, expected_revision: settings.revision, expected_user_id: owner } : {
      provider, ...(action !== 'models' ? { model: model.trim() } : {}), ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
      ...(action === 'save' ? { expected_revision: settings.revision } : {}), expected_user_id: owner,
    };
    try {
      const result = await request(`account/ai-settings${action === 'save' ? '' : `/${action}`}`, body);
      if (!current(epoch)) return;
      if (!record(result) || result.user_id !== owner) throw Error('invalid_response');
      if (action === 'models') {
        if (!Array.isArray(result.models) || result.models.length > 2000) throw Error('invalid_response');
        const rows = result.models.map<Model>(item => { if (!record(item) || !modelId(item.id) || typeof item.name !== 'string' || item.name.length > 300) throw Error('invalid_response'); return { id: item.id, name: item.name }; });
        setModels(rows); setCustomModel(false); setNotice(rows.length ? t('模型列表已更新', 'Model list updated') : t('没有可用模型，请填写模型 ID。', 'No models returned. Enter a model ID.'));
      } else if (action === 'test') {
        if (result.ok !== true || result.provider !== provider || result.model !== model.trim()) throw Error('invalid_response');
         setNotice(t('连接成功', 'Connection successful'));
      } else {
        const next = readSettings(result, owner);
        if (next.revision !== settings.revision + 1) throw Error('invalid_response');
        apply(next, provider); setNotice(action === 'remove' ? t('密钥已删除', 'Key deleted') : t('已保存并启用', 'Saved and active')); onChanged?.();
      }
    } catch (reason) {
      if (current(epoch)) {
        setError(message(reason));
        const status = record(reason) && typeof reason.status === 'number' ? reason.status : 0;
        if ((action === 'save' || action === 'remove') && (!status || status >= 500 || status === 409)) setBlocked(true);
      }
    } finally { if (current(epoch)) { pending.current = false; setBusy(''); } }
  }
  return <div className="ai-settings-form">
    {error && <p className="ai-settings-error" role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    {busy === 'load' && <p role="status">{t('正在读取 AI 设置…', 'Loading AI settings…')}</p>}
    {settings && <>
      <label><span>{t('服务商', 'Provider')}</span><select aria-label={t('AI 服务商', 'AI provider')} value={provider} disabled={!!busy} onChange={event => choose(event.target.value as Provider)}>{providers.map(item => <option key={item.id} value={item.id}>{item.label}{settings.active_provider === item.id ? ` · ${t('正在使用', 'Active')}` : settings.providers.find(row => row.id === item.id)?.configured ? ` · ${t('已保存', 'Saved')}` : ''}</option>)}</select></label>
      <div className="ai-settings-state">{settings.active_provider === provider && <span className="ai-settings-active">{t('正在使用', 'Active')}</span>}{selected?.configured && <span>{t('密钥已保存', 'Key saved')}</span>}</div>
      {!settings.available && <p role="status">{t('个人 AI 设置暂不可用。', 'Personal AI settings are unavailable.')}</p>}
      <label><span>API key</span><input ref={keyInput} aria-label="API key" type="password" autoComplete="off" spellCheck={false} autoCapitalize="off" maxLength={8192} disabled={!enabled} value={apiKey} placeholder={selected?.configured ? t('留空以使用已保存密钥', 'Leave blank to use the saved key') : ''} onChange={event => { setApiKey(event.target.value); setModels([]); setNotice(''); }} /></label>
      <div className="ai-settings-models"><label><span>{models.length ? t('模型', 'Model') : t('模型 ID', 'Model ID')}</span>{models.length ? <select aria-label={t('AI 模型', 'AI model')} disabled={!enabled} value={manualModel ? '__custom__' : model} onChange={event => { setCustomModel(event.target.value === '__custom__'); if (event.target.value !== '__custom__') setModel(event.target.value); setNotice(''); }}><option value="">{t('选择模型', 'Choose a model')}</option>{models.map(item => <option value={item.id} key={item.id}>{item.name}</option>)}<option value="__custom__">{t('自定义模型…', 'Custom model…')}</option></select> : <input aria-label={t('模型 ID', 'Model ID')} autoComplete="off" spellCheck={false} maxLength={200} disabled={!enabled} value={model} onChange={event => { setModel(event.target.value); setNotice(''); }} />}</label><button type="button" disabled={!enabled || !hasKey} onClick={() => void perform('models')}>{busy === 'models' ? t('读取中…', 'Loading…') : t('读取模型', 'Load models')}</button></div>
      {models.length > 0 && manualModel && <label><span>{t('模型 ID', 'Model ID')}</span><input aria-label={t('模型 ID', 'Model ID')} autoComplete="off" spellCheck={false} maxLength={200} disabled={!enabled} value={model} onChange={event => { setModel(event.target.value); setNotice(''); }} /></label>}
      <div className="ai-settings-actions"><button type="button" disabled={!enabled || !hasKey || !modelId(model.trim())} onClick={() => void perform('test')}>{busy === 'test' ? t('测试中…', 'Testing…') : t('测试连接', 'Test connection')}</button><button type="button" className="ai-settings-primary" disabled={!enabled || !hasKey || !modelId(model.trim())} onClick={() => void perform('save')}>{busy === 'save' ? t('保存中…', 'Saving…') : t('保存并使用', 'Save and use')}</button>{selected?.configured && <button type="button" className="ai-settings-delete" disabled={!!busy || blocked} onClick={() => void perform('remove')}>{t('删除密钥', 'Delete key')}</button>}</div>
      <p className="ai-settings-privacy">{t('密钥在账号内加密保存。AI 操作会将所选内容发送给此服务商，用量由服务商计费。', 'Your key is encrypted in your account. AI actions send selected content to this provider; usage is billed by the provider.')}</p>
    </>}
    {(!settings || blocked) && !busy && <button type="button" onClick={() => void load()}>{t('重新读取设置', 'Reload settings')}</button>}
  </div>;
}
