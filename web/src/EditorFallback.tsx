import { FeatureFailure, FeatureLoading } from './deferredFeature';
import { useI18n } from './i18n';

export type PythonEditorProps = { value: string; readOnly: boolean; theme: 'light' | 'dark'; font: number; label: string; onChange: (value: string) => void };
const label: [string, string] = ['代码编辑器', 'code editor'];
export function EditorLoading({ value, font }: PythonEditorProps) {
  return <div className="editor-fallback" aria-busy="true"><FeatureLoading label={label} /><pre style={{ fontSize: font }}>{value}</pre></div>;
}
export function EditorFailure({ value, readOnly, font, label: accessibleLabel, onChange, retry }: PythonEditorProps & { retry: () => void }) {
  const { t } = useI18n();
  return <div className="editor-fallback">
    <FeatureFailure label={label} retry={retry}><small>{t('已切换为普通文本模式，可以继续编辑、保存和运行。', 'Plain-text mode is available. You can keep editing, saving and running your code.')}</small></FeatureFailure>
    <textarea aria-label={accessibleLabel} value={value} readOnly={readOnly} onChange={event => onChange(event.target.value)} spellCheck={false} wrap="off" style={{ fontSize: font }} />
  </div>;
}
