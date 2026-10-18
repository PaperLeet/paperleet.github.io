import {useRef,useState} from 'react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import {useAccount} from './social';
import CommunityAuthor from './CommunityAuthor';
import {communityPostHref} from './communityNavigation';
import {postPreview} from './postExcerpt';
import {useAdminDraft,AdminDraftActions} from './AdminDrafts';
import {ReadStatus,useAdminRead} from './AdminParts';
import {highlightPostId,isHighlightPost,isAdminHighlights,type AdminHighlightList,type HighlightSlot,type HighlightPost} from './communityHighlightsModel';
import type {AdminListProps} from './AdminLists';
import './community-highlights.css';

export default function AdminHighlights({owner,view,epoch,run,blocked,onChange}:AdminListProps){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy(),category=view.board||'',{capabilities}=useAccount();
 const result=useAdminRead<AdminHighlightList>('admin/highlights'+(category?'?category='+encodeURIComponent(category):''),owner,epoch,value=>isAdminHighlights(value,category),capabilities?.admin_highlights===true);
 if(capabilities?.admin_highlights!==true)return <p className="admin-stale">{t('当前服务尚未提供精选管理。','Recommendations are unavailable on this service.')} <AppLink href="#/community/admin">{t('返回概览','Back to overview')}</AppLink></p>;
 return <><p className="admin-intro">{t('首页与每个版位各有 3 个精选位。预览并核对内容后保存；帖子修改后需重新预览；原帖不存在、作者停用或所属版位停用期间不展示。','The homepage and each board have three recommendation slots. Preview and review before saving. Edited posts need a fresh preview. Missing posts, disabled authors and paused boards are not displayed.')}</p>
  <label className="admin-highlight-scope">{t('展示位置','Placement')}<select value={category} onChange={e=>onChange({...view,board:e.target.value||undefined})}><option value="">{t('社区首页','Community homepage')}</option>{taxonomy.data.boards.map(board=><option key={board.id} value={board.id}>{t(board.zh,board.en)}{!board.enabled?t('（已停用）',' (paused)'):''}</option>)}</select></label><ReadStatus {...result}/>
  {result.data&&<div className="admin-highlight-slots">{result.data.slots.map(row=><HighlightForm key={row.id} row={row} category={category} owner={owner} run={run} blocked={blocked||result.loading||!!result.error}/>)}</div>}
 </>;
}
function HighlightForm({row,category,owner,run,blocked}:{row:HighlightSlot;category:string}&Pick<AdminListProps,'owner'|'run'|'blocked'>){
 const {t}=useI18n(),draft=useAdminDraft('highlights:'+row.id,{base:row,input:row.post_id||'',reason:'',candidate:undefined as HighlightPost|undefined},t(`精选位 ${row.slot}`,`Recommendation ${row.slot}`)),{base,input,reason,candidate}=draft.value;
 const [previewId,setPreviewId]=useState(''),[previewRetry,setPreviewRetry]=useState(0),root=useRef<HTMLElement>(null),stale=base.revision!==row.revision;
 const preview=useAdminRead<{owner_id:string;category:string;post:HighlightPost}>(`admin/highlights/preview?category=${encodeURIComponent(category)}&post_id=${previewId}`,owner,previewRetry,value=>value.category===category&&isHighlightPost(value.post,category)&&value.post.id===previewId,!!previewId);
 const shown=preview.data?.post||candidate,id=highlightPostId(input),ready=!!shown&&shown.id===id&&!stale&&!blocked&&!preview.loading&&!preview.error;
 function submit(clear=false){if(blocked||stale||!reason.trim()||!clear&&!ready)return;void run('admin/highlights','highlights',base,clear?t('撤下精选','Remove recommendation'):t('设置精选','Set recommendation'),reason,{category,slot:row.slot,post_id:clear?null:shown!.id,expected_content_version:clear?null:shown!.content_version},draft.intent());}
 const state={empty:t('空位','Empty'),active:t('正在展示','Visible'),changed:t('内容已变更，待复核','Changed; review needed'),unavailable:t('帖子或版位暂不可用','Post or board unavailable')}[row.state];
 return <section ref={root} className="admin-card admin-highlight-slot"><header><h2>{t(`精选位 ${row.slot}`,`Recommendation ${row.slot}`)}</h2><span className="admin-taxonomy-badge" data-state={row.state==='active'?'active':'paused'}>{state}</span></header>
  {row.post&&<p>{t('当前关联：','Currently selected: ')}<AppLink href={communityPostHref(row.post.id)} target="_blank" rel="noopener noreferrer">{row.post.title} ↗</AppLink></p>}{row.state==='unavailable'&&row.post_id&&<code>{row.post_id}</code>}
  {stale&&<p role="status">{t('此精选位已更新，输入已保留。核对最新状态后再继续。','This slot changed. Your inputs are kept; review the latest state.')} <button type="button" className="text-button" onClick={()=>{draft.change({base:row,candidate:undefined});setPreviewId('');}}>{t('已核对，使用最新状态','Reviewed; use latest state')}</button></p>}
  <form className="admin-form" onSubmit={e=>{e.preventDefault();submit();}}><label>{t('帖子链接或 ID','Post link or ID')}<input maxLength={1800} value={input} onChange={e=>{if(preview.data)draft.change({input:e.target.value,candidate:preview.data.post});else draft.change({input:e.target.value});setPreviewId('');}}/></label>
   <button type="button" className="button subtle" disabled={blocked||!id||preview.loading} onClick={()=>{setPreviewId(id!);setPreviewRetry(n=>n+1);}}>{t('预览当前帖子','Preview current post')}</button>{previewId&&<ReadStatus {...preview}/>}
   {shown&&shown.id===id&&<div className="admin-highlight-preview"><b>{shown.title}</b><p>{postPreview(shown.body_excerpt,320,t('[公式]','[formula]')).text}</p><CommunityAuthor user={shown.author} compact/><AppLink href={communityPostHref(shown.id)} target="_blank" rel="noopener noreferrer">{t('打开完整帖子核对','Review the full post')} ↗</AppLink></div>}
   <label>{t('操作原因（仅管理员可见）','Reason (administrator-only)')}<textarea maxLength={1000} required rows={2} value={reason} onChange={e=>draft.change({reason:e.target.value,...(preview.data?{candidate:preview.data.post}:{})})}/></label>
   <div className="admin-highlight-actions"><button className="button" disabled={!ready||!reason.trim()}>{t('保存已预览的版本','Save previewed version')}</button>{row.post_id&&<button type="button" className="text-button" disabled={blocked||stale||!reason.trim()} onClick={()=>submit(true)}>{t('撤下此精选','Remove recommendation')}</button>}</div><AdminDraftActions draft={draft} disabled={blocked} onReset={()=>setPreviewId('')}/>
  </form>
 </section>;
}
