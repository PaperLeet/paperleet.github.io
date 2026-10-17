import { useEffect, useId, useRef, useState } from 'react';
import CommunityMarkdown from './CommunityMarkdown';
import MarkdownEditor from './MarkdownEditor';
import { useI18n } from './i18n';
import { SocialError, SocialNotice, errorMessage, socialApi, useAccount } from './social';
import type { Post } from './communityDiscussion';
import { tabDraftStorage } from './communityDrafts';
import { canEditPost, clearSavedPostEdit, newPostEdit, readPostEdit, writePostEdit, type PostEditDraft } from './communityEdits';
import InterviewFields,{type InterviewModule} from './InterviewFields';
import {loadInterviewModules} from './interviewApi';
import InterviewMeta from './InterviewMeta';
import PaperLinkField from './PaperLinkField';
import PaperPreviewCard from './PaperPreviewCard';
import { canonicalArxivUrl } from './communityPapers';
import {affiliationRequired,canPublishCommunity} from './communityPublishing';
import PublishingAffiliationNotice from './PublishingAffiliationNotice';
import { communityCategories, communityCategoryHint } from './communityCategories';
import {topicFromProblem} from './communityTopics';
import {communityPostHref} from './communityNavigation';

export default function PostEditor({post,owner,disabled,onSaved,onClose,onRefresh}:{
  post:Post;owner:string;disabled:boolean;onSaved:(post:Post)=>void;onClose:()=>void;onRefresh:()=>Promise<boolean>;
}) {
  const {t}=useI18n();
  const {user,capabilities,loading:accountLoading,refresh:refreshAccount}=useAccount();
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading;
  const [draft,setDraft]=useState(()=>readPostEdit(tabDraftStorage(),owner,post.id)||newPostEdit(post));
  const current=useRef(draft), mounted=useRef(true), saving=useRef(false);
  const [busy,setBusy]=useState(false), [error,setError]=useState(''), [saved,setSaved]=useState<boolean|null>(null);
  const [preview,setPreview]=useState(false), [discard,setDiscard]=useState(false);
  const [modules,setModules]=useState<InterviewModule[]>([]),[catalogError,setCatalogError]=useState(false),[catalogRetry,setCatalogRetry]=useState(0);
  useEffect(()=>{
    if(!post.interview)return;
    let live=true;
    void loadInterviewModules().then(value=>{if(live){setModules(value);setCatalogError(false);}}).catch(()=>{if(live)setCatalogError(true);});
    return()=>{live=false;};
  },[!!post.interview,catalogRetry]);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
  const needsReview=draft.baseVersion!==post.edit_version;
  const currentCategory=communityCategories.find(category=>category.id===post.category);
  const categoryHint=communityCategoryHint(draft.category),categoryHintId=useId();
  function edit(change:Partial<PostEditDraft>) {
    const next={...current.current,...change}; current.current=next;setDraft(next);
    setSaved(writePostEdit(tabDraftStorage(),owner,post.id,next));
  }
  async function save() {
    if(busy||disabled||!canPublish||saving.current||needsReview||!canEditPost(post,owner)||!draft.title.trim()||!draft.body.trim())return;
    if(draft.category==='papers' && draft.paperUrl?.trim() && (!canonicalArxivUrl(draft.paperUrl) || !capabilities?.community_papers)) {
      setError(!capabilities?.community_papers ? t('论文链接暂时无法保存，请保留修改草稿并稍后重试。','Paper links cannot be saved right now. Keep your edit draft and try again shortly.') : t('请添加有效的 arXiv 论文链接，或清空此字段。','Add a valid arXiv paper link, or clear the field.'));
      return;
    }
    const submitted={...current.current};saving.current=true;setBusy(true);setError('');
    try {
      const confirmed=await socialApi<Post>(`community/posts/${encodeURIComponent(post.id)}/edit`,{
        title:submitted.title,body:submitted.body,category:submitted.category,
        expected_version:submitted.baseVersion,expected_user_id:owner,
        ...(submitted.interview?{interview:submitted.interview}:{}),
        ...((submitted.paperUrl !== undefined || post.paper_url) ? {paper_url:submitted.category==='papers' ? canonicalArxivUrl(submitted.paperUrl) || null : null} : {}),
      });
      clearSavedPostEdit(tabDraftStorage(),owner,post.id,submitted);
      if(mounted.current)onSaved(confirmed);
    } catch(e) {
      if(mounted.current) {
        setError(e instanceof SocialError && e.code==='edit_conflict'
          ? t('帖子已被更新，修改草稿已保留。请核对最新内容后继续。','The post has changed. Your edit draft is kept; review the latest content before continuing.')
          :errorMessage(e));
        if(e instanceof SocialError && e.code==='edit_conflict')await onRefresh();
        if(affiliationRequired(e))void refreshAccount().catch(()=>{});
      }
    } finally {saving.current=false;if(mounted.current)setBusy(false);}
  }
  return <section className="social-panel community-composer" aria-label={t('编辑帖子','Edit post')}>
    <div className="panel-title"><h2>{t('编辑帖子','Edit post')}</h2><button type="button" disabled={busy} onClick={onClose}>{t('关闭编辑','Close editor')}</button></div>
    {error&&<SocialNotice error={error}/>}
    <PublishingAffiliationNotice/>
    <p className="community-draft-status" role="status">{saved===false
      ?t('此标签页无法保存修改草稿，关闭或离开前请复制内容。','This tab cannot save your edit draft. Copy the text before closing or leaving.')
      :t('修改草稿按账号和帖子保存在本标签页；保存修改后才会公开更新。','Edit drafts stay in this tab, separately for your account and post. Changes become public only when saved.')}</p>
    {needsReview&&<div className="publication-recovery" role="status">
      <p>{t('当前草稿基于较早版本。请对照最新帖子，将需要保留的内容合并到草稿。','This draft is based on an older version. Compare the latest post and merge anything you want to keep into your draft.')}</p>
      <details open><summary>{t('最新帖子内容','Latest post content')}</summary><strong>{post.title}</strong>{post.interview&&<InterviewMeta details={post.interview} topics={modules.map(topicFromProblem)} catalogLoading={!modules.length&&!catalogError} returnTo={communityPostHref(post.id)}/>}<CommunityMarkdown body={post.body}/><PaperPreviewCard url={post.paper_url} paper={post.paper}/><p>{t('版块','Category')}: {currentCategory?t(currentCategory.zh,currentCategory.en):post.category}</p></details>
      <button type="button" disabled={busy||disabled} onClick={()=>{edit({baseVersion:post.edit_version!});setError('');}}>{t('已核对，以此版本继续编辑','Reviewed; continue from this version')}</button>
    </div>}
    <form className="social-form" onSubmit={e=>{e.preventDefault();void save();}}>
      {post.interview&&<InterviewFields value={draft.interview||post.interview} onChange={interview=>edit({interview,category:'career'})} problems={modules} disabled={busy}/>}
      {catalogError&&<p role="status">{t('暂时无法读取关联练习，现有关联会保留。','Related modules could not load. Existing links are preserved.')} <button type="button" onClick={()=>setCatalogRetry(value=>value+1)}>{t('重试加载','Retry loading')}</button></p>}
      <label>{t('版块','Category')}<select value={draft.category} disabled={busy||!!post.interview||post.solution===true} aria-describedby={categoryHint?categoryHintId:undefined} onChange={e=>edit({category:e.target.value})}>
        {communityCategories.map(({id,zh,en})=><option key={id} value={id}>{t(zh,en)}</option>)}
      </select>{categoryHint&&<small id={categoryHintId}>{t(...categoryHint)}</small>}</label>
      {draft.category==='papers' && <PaperLinkField value={draft.paperUrl || ''} onChange={paperUrl=>edit({paperUrl})} owner={owner} available={capabilities?.community_papers===true} disabled={busy} initialPaper={post.paper} onUseTitle={title=>edit({title:title.slice(0,180)})}/>}
      <label>{t('标题','Title')}<input autoFocus required maxLength={180} value={draft.title} disabled={busy} onChange={e=>edit({title:e.target.value})}/></label>
      <MarkdownEditor scopeKey={JSON.stringify(['post-edit',owner,post.id])} submitShortcut value={draft.body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
        label={t('正文','Post body')} modeLabel={t('帖子修改模式','Post edit mode')} rows={10} maxLength={20000} busy={busy} disabled={disabled}/>
      <div className="form-footer"><button type="button" className="text-button" disabled={busy} onClick={()=>setDiscard(true)}>{t('丢弃修改','Discard edits')}</button><button className="button" disabled={busy||disabled||!canPublish||needsReview||!draft.title.trim()||!draft.body.trim()}>{busy?t('正在保存…','Saving…'):t('保存修改','Save changes')}</button></div>
    </form>
    {discard&&<div className="draft-discard-confirm" role="alert"><p>{t('丢弃未保存的修改？已发布的帖子不会改变。','Discard unsaved edits? The published post will stay as it is.')}</p><button type="button" disabled={busy} onClick={()=>{clearSavedPostEdit(tabDraftStorage(),owner,post.id,current.current);onClose();}}>{t('确认丢弃修改','Discard unsaved edits')}</button><button type="button" onClick={()=>setDiscard(false)}>{t('保留修改','Keep edits')}</button></div>}
  </section>;
}
