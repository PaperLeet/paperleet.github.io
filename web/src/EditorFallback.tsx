import { FeatureFailure, FeatureLoading } from './deferredFeature';
import { useI18n } from './i18n';

export type EditorRecovery = { blocked: boolean; reason: string; reload: () => void };
export type PythonEditorProps = { value: string; readOnly: boolean; theme: 'light' | 'dark'; font: number; label: string; onChange: (value: string) => void; recovery?: EditorRecovery };
const label: [string, string] = ['代码编辑器', 'code editor'];
export function EditorLoading({ value, font }: PythonEditorProps) {
  return <div className="editor-fallback" aria-busy="true"><FeatureLoading label={label} /><pre style={{ fontSize: font }}>{value}</pre></div>;
}
export function EditorFailure({ value, readOnly, font, label: accessibleLabel, onChange, retry, recovery }: PythonEditorProps & { retry: () => void }) {
  const { t } = useI18n();
  return <div className="editor-fallback">
    {recovery ? <div className="feature-failure editor-recovery-notice">
      <p role="alert">{t('代码编辑器暂时无法加载。', 'The code editor could not load.')}</p>
      <p>{t('可以继续用普通文本模式编辑和运行。重新加载会刷新此页；代码和已保存的测试结果会保留。',
        'You can keep editing and running in plain-text mode. Reloading refreshes this page; your code and saved test results are kept.')}</p>
      <button className="button subtle" type="button" disabled={recovery.blocked} onClick={recovery.reload}>{t('重新加载编辑器', 'Reload editor')}</button>
      {recovery.reason && <p role="status">{recovery.reason}</p>}
    </div> : <FeatureFailure label={label} retry={retry}><small>{t('已切换为普通文本模式，可以继续编辑、保存和运行。', 'Plain-text mode is available. You can keep editing, saving and running your code.')}</small></FeatureFailure>}
    <textarea aria-label={accessibleLabel} value={value} readOnly={readOnly} onChange={event => onChange(event.target.value)} spellCheck={false} wrap="off" style={{ fontSize: font }} />
  </div>;
}
