import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import CommunityMarkdown from './CommunityMarkdown';
import MarkdownEditor from './MarkdownEditor';
import { useI18n } from './i18n';
import { SocialError, SocialNotice, errorMessage, socialApi, useAccount } from './social';
import type { Reply } from './communityDiscussion';
import { tabDraftStorage } from './communityDrafts';
import { canEditReply, confirmedReplyEdit, clearSavedReplyEdit, newReplyEdit, readReplyEdit, writeReplyEdit, type ReplyEditDraft,type ReplyEditOrigin,type ReplyEditConfirmation } from './replyEdits';
import {affiliationRequired,canPublishCommunity} from './communityPublishing';
import PublishingAffiliationNotice from './PublishingAffiliationNotice';
import {signInHref} from './accountNavigation';
import {keepReadingTriggerVisible} from './readingViewport';


export default function ReplyEditor({reply,owner,disabled,onSaved,onClose,onRefresh,focusOrigin,compact=false}:{
  reply:Reply;owner:string;disabled:boolean;onSaved:(reply:Reply,confirmation?:ReplyEditConfirmation)=>void;onClose:()=>void;onRefresh:()=>Promise<boolean>;focusOrigin?:ReplyEditOrigin;compact?:boolean;
}) {
  const {t}=useI18n();
  const {user,capabilities,loading:accountLoading,refresh:refreshAccount}=useAccount();
  const [accountUnconfirmed,setAccountUnconfirmed]=useState(false);
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading&&user?.id===owner&&!accountUnconfirmed;
  const [draft,setDraft]=useState(()=>readReplyEdit(tabDraftStorage(),owner,reply.id)||newReplyEdit(reply));
  const current=useRef(draft), mounted=useRef(true), saving=useRef(false);
  const [busy,setBusy]=useState(false), [error,setError]=useState(''), [saved,setSaved]=useState<boolean|null>(null);
  const field=useRef<HTMLTextAreaElement>(null),entry=useRef(focusOrigin);
  const [notice,setNotice]=useState('');
  useLayoutEffect(()=>{
    const intent=entry.current;if(!intent||disabled||accountLoading||busy)return;entry.current=undefined;
    if(location.href!==intent.href)return;
    const active=document.activeElement,opener=intent.opener;
    if(active!==opener&&!(active===document.body&&opener instanceof HTMLElement&&(!opener.isConnected||opener.matches(':disabled')||!!opener.closest('[hidden]'))))return;
    const node=field.current;if(!node||node.value!==current.current.body||node.closest('[hidden]'))return;
    node.focus({preventScroll:true});keepReadingTriggerVisible(node);
  },[disabled,accountLoading,busy]);
  const [preview,setPreview]=useState(false), [discard,setDiscard]=useState(false);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
  const needsReview=draft.baseVersion!==reply.edit_version;
  function edit(change:Partial<ReplyEditDraft>) {
    const next={...current.current,...change}; current.current=next;setDraft(next);
    setSaved(writeReplyEdit(tabDraftStorage(),owner,reply.id,next));setNotice('');
  }
  async function save() {
    if(busy||disabled||!canPublish||saving.current||needsReview||!canEditReply(reply,owner)||!draft.body.trim())return;
    const submitted={...current.current},origin={opener:document.activeElement,href:location.href};saving.current=true;setBusy(true);setError('');setNotice('');
    try {
      const response=await socialApi<Reply>(`community/replies/${encodeURIComponent(reply.id)}/edit`,{
        body:submitted.body,expected_version:submitted.baseVersion,expected_user_id:owner,
      },AbortSignal.timeout(20_000));
      const confirmed=confirmedReplyEdit(reply,owner,submitted,response);
      if(!confirmed)throw Error(t('修改尚未确认，草稿已保留。请重试或核对最新内容。','The edit is not confirmed. Your draft is kept; retry or review the latest content.'));
      clearSavedReplyEdit(tabDraftStorage(),owner,reply.id,submitted);
      if(mounted.current){
        const keepEditing=current.current.body!==submitted.body||current.current.baseVersion!==submitted.baseVersion;
        onSaved(confirmed,{origin,baseVersion:submitted.baseVersion,keepEditing});
        if(keepEditing)setNotice(t('先前修改已保存，后续修改仍在草稿中。请核对最新版本后继续。','The earlier edit is saved. Your later changes remain in the draft; review the latest version to continue.'));
      }
    } catch(e) {
      if(mounted.current) {
        setError(e instanceof SocialError && e.code==='edit_conflict'
          ? t('回复已被更新，修改草稿已保留。请核对最新内容后继续。','The reply has changed. Your edit draft is kept; review the latest content before continuing.')
          :errorMessage(e));
        if(e instanceof Error&&(e.name==='TimeoutError'||e.name==='AbortError'))setError(t('保存超时，修改草稿已保留。可重试或核对最新内容。','Saving timed out. Your edit draft is kept; retry or review the latest content.'));
        if(e instanceof SocialError&&(e.status===401||e.code==='account_changed')){
          setAccountUnconfirmed(true);try{await refreshAccount();if(mounted.current)setAccountUnconfirmed(false);}catch{/* Explicit recovery keeps the private edit. */}
        }
        if(e instanceof SocialError && e.code==='edit_conflict')await onRefresh();
        if(affiliationRequired(e))void refreshAccount().catch(()=>{});
      }
    } finally {saving.current=false;if(mounted.current)setBusy(false);}
  }
  return <section className={'social-panel community-composer reply-editor'+(compact?' is-compact':'')} aria-label={compact?t('编辑评论','Edit comment'):t('编辑回复','Edit reply')}>
    <div className="panel-title">{compact?<h3>{t('编辑评论','Edit comment')}</h3>:<h2>{t('编辑回复','Edit reply')}</h2>}<button type="button" disabled={busy} onClick={onClose}>{t('关闭编辑','Close editor')}</button></div>
    {error&&<SocialNotice error={error}/>}
    {notice&&<p role="status">{notice}</p>}
    {accountUnconfirmed&&<p role="status">{t('账号尚未确认，修改草稿已保留。','Your account is unconfirmed. Your edit draft is kept.')} <button type="button" className="text-button" disabled={busy||accountLoading} onClick={()=>{void refreshAccount().then(()=>{if(mounted.current)setAccountUnconfirmed(false);}).catch(()=>{});}}>{t('刷新账号','Refresh account')}</button><a href={signInHref(typeof location==='undefined'?'community':location.hash)}>{t('重新登录','Sign in again')}</a></p>}
    <PublishingAffiliationNotice/>
    <p className="community-draft-status" role="status">{saved===false
      ?t('此标签页无法保存修改草稿，关闭或离开前请复制内容。','This tab cannot save your edit draft. Copy the text before closing or leaving.')
      :t('修改草稿按账号和回复保存在本标签页；保存修改后才会公开更新。','Edit drafts stay in this tab, separately for your account and reply. Changes become public only when saved.')}</p>
    {needsReview&&<div className="publication-recovery" role="status">
      <p>{t('当前草稿基于较早版本。请对照最新回复，将需要保留的内容合并到草稿。','This draft is based on an older version. Compare the latest reply and merge anything you want to keep into your draft.')}</p>
      <details open><summary>{t('最新回复内容','Latest reply content')}</summary><CommunityMarkdown body={reply.body}/></details>
      <button type="button" disabled={busy||disabled} onClick={()=>{edit({baseVersion:reply.edit_version!});setError('');}}>{t('已核对，以此版本继续编辑','Reviewed; continue from this version')}</button>
    </div>}
    <form className="social-form" onSubmit={e=>{e.preventDefault();void save();}}>
      <MarkdownEditor scopeKey={JSON.stringify(['reply-edit',owner,reply.id])} submitShortcut value={draft.body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
        label={compact?t('评论内容','Comment body'):t('正文','Reply body')} modeLabel={t('回复修改模式','Reply edit mode')} rows={compact?4:6} maxLength={10000} busy={busy} disabled={disabled||accountLoading||accountUnconfirmed||user?.id!==owner} textareaRef={field}/>
      <div className="form-footer"><button type="button" className="text-button" disabled={busy} onClick={()=>setDiscard(true)}>{t('丢弃修改','Discard edits')}</button><button className="button" disabled={busy||disabled||!canPublish||needsReview||!draft.body.trim()}>{busy?t('正在保存…','Saving…'):t('保存修改','Save changes')}</button></div>
    </form>
    {discard&&<div className="draft-discard-confirm" role="alert"><p>{t('丢弃未保存的修改？已发布的回复不会改变。','Discard unsaved edits? The published reply will stay as it is.')}</p><button type="button" disabled={busy} onClick={()=>{clearSavedReplyEdit(tabDraftStorage(),owner,reply.id,current.current);onClose();}}>{t('确认丢弃修改','Discard unsaved edits')}</button><button type="button" onClick={()=>setDiscard(false)}>{t('保留修改','Keep edits')}</button></div>}
  </section>;
}
