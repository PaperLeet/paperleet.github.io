import type { BrowserSyntaxFailure } from './browserSyntax';
import { useI18n } from './i18n';
import { SourceLocationButton } from './SourceLocationButton';
import type { SourceLocation } from './sourceLocation';

export function BrowserSyntaxNotice({failure,source,onLocate}:{failure:BrowserSyntaxFailure;source:string;onLocate?:(location:SourceLocation)=>void}) {
  const {t}=useI18n();
  const current=failure.source===source;
  const {kind,message,line,column}=failure.diagnostic;
  return <section className="browser-syntax-notice" role={current?'alert':'status'} aria-label={t('Python 语法检查','Python syntax check')}>
    <strong>{current?t('Python 语法检查失败','Python syntax check failed'):t('上次语法检查失败 · 代码已修改','Previous syntax check failed · code changed')}</strong>
    <p>{t(`第 ${line} 行，第 ${column} 列`,`Line ${line}, column ${column}`)}</p>
    <pre>{kind}: {message}</pre>
    {current && <SourceLocationButton location={{line,column}} onLocate={onLocate} />}
    <p>{current?t('仅完成语法检查。修正后重新运行，才能执行完整校准与测试。','Only syntax was checked. Fix this error and run again to start full calibration and testing.'):t('此定位对应上次运行的代码，请重新检查当前实现。','This location belongs to the previous source. Run again to check your current implementation.')}</p>
  </section>;
}
