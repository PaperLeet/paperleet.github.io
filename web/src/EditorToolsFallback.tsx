import {useI18n} from './i18n';

export default function EditorToolsFallback({retry}:{retry?:()=>void}){
  const {t}=useI18n();
  return <span className="editor-tools-recovery">
    <span role={retry?'alert':'status'}>{retry?t('编辑工具暂不可用','Editor tools unavailable'):t('正在加载编辑工具…','Loading editor tools…')}</span>
    {retry&&<button type="button" className="text-button" onClick={event=>{
      const target=event.currentTarget.closest('.editor-tools')?.querySelector<HTMLButtonElement>('.editor-download-code:not(:disabled)');
      target?.focus({preventScroll:true});retry();
    }}>{t('重试工具','Retry tools')}</button>}
  </span>;
}
