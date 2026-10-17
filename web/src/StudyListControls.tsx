import {useState} from 'react';
import {Plus} from 'lucide-react';
import {useI18n} from './i18n';
import {newStudyListId,type StudyListsState,type createStudyListStore,STUDY_LIST_LIMIT} from './studyListStore';
type Store=ReturnType<typeof createStudyListStore>;
export function StudyListFeedback({state,store,focus}:{state:StudyListsState;store?:Store;focus?:()=>void}){
 const {t}=useI18n(),pending=state.pending;
 const action=(event:React.MouseEvent<HTMLButtonElement>,run:()=>unknown)=>{if(document.activeElement===event.currentTarget)focus?.();void run();};
 return <>
  {state.error&&<div className="study-list-feedback" role="status"><p>{t('题单暂时无法读取，已保留当前内容。','Study lists could not load. Current content is kept.')} {state.error}</p><button type="button" className="text-button" disabled={state.loading||pending?.busy} onClick={event=>action(event,()=>store?.refresh())}>{t('重试题单加载','Retry study lists')}</button></div>}
  {pending?.busy&&<p role="status">{t('正在同步题单…','Syncing study lists…')}</p>}
  {pending?.error&&<div className="study-list-feedback" role="status"><p>{pending.conflict?t('另一页面或设备修改了题单。请先载入最新内容；输入会保留，之后可重新提交。','Another page or device changed your lists. Load the latest version; your input is kept for resubmission.'):pending.rejected?t('操作未应用。可返回修改后重新提交。','The change was not applied. Return to editing and try again.'):t('操作结果尚未确认。请重试确认，以免重复创建或改动。','The result is unconfirmed. Retry confirmation to avoid duplicate changes.')} {pending.error}</p><button type="button" className="text-button" onClick={event=>action(event,()=>pending.conflict?store?.loadLatest():pending.rejected?store?.dismissRejected():store?.retry())}>{pending.conflict?t('载入最新题单','Load latest lists'):pending.rejected?t('返回修改','Return to editing'):t('重试确认','Retry confirmation')}</button></div>}
 </>;
}
export function CreateStudyList({state,store,problemId}:{state:StudyListsState;store?:Store;problemId?:string}){
 const {t}=useI18n(),[title,setTitle]=useState('');
 const blocked=!!state.pending||state.loading||!!state.error||!state.snapshot||state.snapshot.lists.length>=STUDY_LIST_LIMIT;
 return <form className="study-list-create" onSubmit={event=>{
  event.preventDefault();if(blocked||!title.trim()||Array.from(title.trim()).length>60)return;
  const sent=title;void store?.mutate({action:'create',list_id:newStudyListId(),title:title.trim(),...(problemId?{problem_ids:[problemId]}:{})}).then(ok=>{if(ok)setTitle(current=>current===sent?'':current);});
 }}><label><span>{t('新题单名称','New study-list name')}</span><input value={title} onChange={event=>setTitle(event.target.value)} maxLength={120} required placeholder={t('例如：面试复习','For example: Interview review')}/></label><button type="submit" className="button subtle" disabled={blocked||!title.trim()||Array.from(title.trim()).length>60}><Plus size={15} aria-hidden="true"/>{problemId?t('新建并加入','Create and add'):t('新建题单','Create list')}</button>{state.snapshot&&state.snapshot.lists.length>=STUDY_LIST_LIMIT&&<small>{t('最多保存 30 个私人题单。','You can keep up to 30 private study lists.')}</small>}</form>;
}
