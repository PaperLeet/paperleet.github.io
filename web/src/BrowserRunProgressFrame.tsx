import type {ReactNode} from 'react';
import {useI18n} from './i18n';
export function BrowserRunProgressFrame({onStop,children}:{onStop:()=>void;children:ReactNode}){
  const {t}=useI18n();
  return <section className="browser-run-progress" aria-label={t('测试进度','Test progress')}>
    <div className="browser-run-progress-header"><strong>{t('正在验证你的实现','Checking your implementation')}</strong><button type="button" className="browser-progress-stop" onClick={onStop}>{t('停止运行','Stop run')}</button></div>
    <div className="browser-run-progress-body">{children}</div>
  </section>;
}
