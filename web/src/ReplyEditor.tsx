import { useEffect, useRef, useState } from 'react';
import CommunityMarkdown from './CommunityMarkdown';
import MarkdownEditor from './MarkdownEditor';
import { useI18n } from './i18n';
import { SocialError, SocialNotice, errorMessage, socialApi } from './social';
import type { Reply } from './communityDiscussion';
import { tabDraftStorage } from './communityDrafts';
import { canEditReply, clearSavedReplyEdit, newReplyEdit, readReplyEdit, writeReplyEdit, type ReplyEditDraft } from './replyEdits';


export default function ReplyEditor({reply,owner,disabled,onSaved,onClose,onRefresh}:{
  reply:Reply;owner:string;disabled:boolean;onSaved:(reply:Reply)=>void;onClose:()=>void;onRefresh:()=>Promise<boolean>;
}) {
  const {t}=useI18n();
  const [draft,setDraft]=useState(()=>readReplyEdit(tabDraftStorage(),owner,reply.id)||newReplyEdit(reply));
  const current=useRef(draft), mounted=useRef(true), saving=useRef(false);
  const [busy,setBusy]=useState(false), [error,setError]=useState(''), [saved,setSaved]=useState<boolean|null>(null);
  const [preview,setPreview]=useState(false), [discard,setDiscard]=useState(false);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
  const needsReview=draft.baseVersion!==reply.edit_version;
  function edit(change:Partial<ReplyEditDraft>) {
    const next={...current.current,...change}; current.current=next;setDraft(next);
    setSaved(writeReplyEdit(tabDraftStorage(),owner,reply.id,next));
  }
  async function save() {
    if(busy||disabled||saving.current||needsReview||!canEditReply(reply,owner)||!draft.body.trim())return;
    const submitted={...current.current};saving.current=true;setBusy(true);setError('');
    try {
      const confirmed=await socialApi<Reply>(`community/replies/${encodeURIComponent(reply.id)}/edit`,{
        body:submitted.body,expected_version:submitted.baseVersion,expected_user_id:owner,
      });
      clearSavedReplyEdit(tabDraftStorage(),owner,reply.id,submitted);
      if(mounted.current)onSaved(confirmed);
    } catch(e) {
      if(mounted.current) {
        setError(e instanceof SocialError && e.code==='edit_conflict'
          ? t('回复已被更新，修改草稿已保留。请核对最新内容后继续。','The reply has changed. Your edit draft is kept; review the latest content before continuing.')
          :errorMessage(e));
        if(e instanceof SocialError && e.code==='edit_conflict')await onRefresh();
      }
    } finally {saving.current=false;if(mounted.current)setBusy(false);}
  }
  return <section className="social-panel community-composer" aria-label={t('编辑回复','Edit reply')}>
    <div className="panel-title"><h2>{t('编辑回复','Edit reply')}</h2><button type="button" disabled={busy} onClick={onClose}>{t('关闭编辑','Close editor')}</button></div>
    {error&&<SocialNotice error={error}/>}
    <p className="community-draft-status" role="status">{saved===false
      ?t('此标签页无法保存修改草稿，关闭或离开前请复制内容。','This tab cannot save your edit draft. Copy the text before closing or leaving.')
      :t('修改草稿按账号和回复保存在本标签页；保存修改后才会公开更新。','Edit drafts stay in this tab, separately for your account and reply. Changes become public only when saved.')}</p>
    {needsReview&&<div className="publication-recovery" role="status">
      <p>{t('当前草稿基于较早版本。请对照最新回复，将需要保留的内容合并到草稿。','This draft is based on an older version. Compare the latest reply and merge anything you want to keep into your draft.')}</p>
      <details open><summary>{t('最新回复内容','Latest reply content')}</summary><CommunityMarkdown body={reply.body}/></details>
      <button type="button" disabled={busy||disabled} onClick={()=>{edit({baseVersion:reply.edit_version!});setError('');}}>{t('已核对，以此版本继续编辑','Reviewed; continue from this version')}</button>
    </div>}
    <form className="social-form" onSubmit={e=>{e.preventDefault();void save();}}>
      <MarkdownEditor value={draft.body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
        label={t('正文','Reply body')} modeLabel={t('回复修改模式','Reply edit mode')} rows={6} maxLength={10000} disabled={busy||disabled} autoFocus/>
      <div className="form-footer"><button type="button" className="text-button" disabled={busy} onClick={()=>setDiscard(true)}>{t('丢弃修改','Discard edits')}</button><button className="button" disabled={busy||disabled||needsReview||!draft.body.trim()}>{busy?t('正在保存…','Saving…'):t('保存修改','Save changes')}</button></div>
    </form>
    {discard&&<div className="draft-discard-confirm" role="alert"><p>{t('丢弃未保存的修改？已发布的回复不会改变。','Discard unsaved edits? The published reply will stay as it is.')}</p><button type="button" disabled={busy} onClick={()=>{clearSavedReplyEdit(tabDraftStorage(),owner,reply.id,current.current);onClose();}}>{t('确认丢弃修改','Discard unsaved edits')}</button><button type="button" onClick={()=>setDiscard(false)}>{t('保留修改','Keep edits')}</button></div>}
  </section>;
}
