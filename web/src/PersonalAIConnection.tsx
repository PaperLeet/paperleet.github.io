import { useI18n } from './i18n';
import { providerLabel } from './personalAI';
import type { PersonalAIStatus } from './personalAI';
import './ai-settings.css';

export default function PersonalAIConnection({ state, onSettings, onRefresh, submission = false }: { state: PersonalAIStatus; onSettings?: () => void; onRefresh?: () => void; submission?: boolean }) {
  const { t } = useI18n();
  return <div className="personal-ai-connection"><span role="status">{state.loading ? t('正在读取 AI 配置…', 'Loading AI configuration…') : state.configured ? `AI · ${providerLabel(state.provider)} · ${state.model}` : state.error ? t('无法读取 AI 配置', 'Could not load AI configuration') : state.available === false ? t('个人 AI 暂不可用', 'Personal AI is unavailable') : submission ? t('提交前请先配置个人 API key', 'Connect your API key before submitting') : t('使用 AI 前请先配置个人 API key', 'Connect your API key to use AI')}</span><div>
    {state.error && onRefresh && <button type="button" onClick={onRefresh}>{t('重试', 'Retry')}</button>}
    {onSettings ? <button type="button" onClick={onSettings}>{t('AI 设置', 'AI settings')}</button> : <span>{t('AI 设置正在更新', 'AI settings are updating')}</span>}
  </div></div>;
}
