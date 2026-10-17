import { useEffect, useRef, useState } from 'react';
import { Bookmark } from 'lucide-react';
import { useI18n } from './i18n';
import { signInHref } from './accountNavigation';
import { socialApi, errorMessage, SocialError, SocialNotice } from './social';
import { confirmedBookmark } from './communityBookmarks';
import './post-bookmark.css';

export default function PostBookmark({postId,owner,bookmarked,enabled,returnTo,disabled,hidden=false,onChange,onBusyChange,onAccountRefresh}:{
  postId:string;owner:string;bookmarked?:boolean;enabled:boolean;returnTo:string;disabled:boolean;hidden?:boolean;
  onChange:(saved:boolean)=>void;onBusyChange:(busy:boolean)=>void;
  onAccountRefresh?:()=>Promise<void>;
}) {
  const {t}=useI18n();
  const [busy,setBusy]=useState(false),[pending,setPending]=useState<boolean>(),[error,setError]=useState('');
  const mounted=useRef(true),working=useRef(false),controller=useRef<AbortController|undefined>(undefined);
  const trigger=useRef<HTMLButtonElement>(null);
  const root=useRef<HTMLDivElement>(null),focusRetained=useRef(false);
  const recovery=useRef<HTMLDivElement>(null);
  const focusConfirmed=useRef(false),focusHref=useRef('');
  const uncertain=pending!==undefined&&pending!==bookmarked;
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;controller.current?.abort();};},[]);
  useEffect(()=>{
    if(hidden||!enabled){focusRetained.current=false;focusConfirmed.current=false;return;}
    if(busy||disabled)return;
    if(focusHref.current!==location.href||(document.activeElement!==document.body&&!root.current?.contains(document.activeElement))){focusRetained.current=false;focusConfirmed.current=false;return;}
    if(error&&uncertain&&focusRetained.current)recovery.current?.querySelector<HTMLButtonElement>('button')?.focus();
    else if(focusConfirmed.current){focusConfirmed.current=false;trigger.current?.focus();}
  },[busy,disabled,hidden,enabled,error,uncertain,bookmarked]);
  async function confirm(active?:boolean) {
    if(hidden||disabled||working.current||!owner||!enabled)return;
    focusRetained.current=!!root.current?.contains(document.activeElement);
    focusHref.current=location.href;focusConfirmed.current=false;
    working.current=true;setBusy(true);onBusyChange(true);setError('');
    const request=new AbortController();controller.current=request;
    const timer=setTimeout(()=>request.abort(),20_000);
    try {
      const value=await socialApi<unknown>(`community/posts/${encodeURIComponent(postId)}/bookmark`,
        active===undefined?undefined:{active,expected_user_id:owner},request.signal);
      const result=confirmedBookmark(value,owner,postId,active);
      if(!result) {
        await onAccountRefresh?.();
        throw Error(t('收藏响应与当前账号或帖子不一致，请重新读取状态。','The saved-state response does not match this account or post. Read the state again.'));
      }
      if(mounted.current) {
        focusConfirmed.current=focusRetained.current;
        setPending(undefined);onChange(result.bookmarked);
      }
    } catch(e) {
      if(mounted.current&&e instanceof SocialError&&(e.status===401||e.code==='account_changed'))await onAccountRefresh?.();
      if(mounted.current) {
        if(active!==undefined)setPending(active);
        setError(request.signal.aborted?t('请求超时，请重试或重新读取收藏状态。','The request timed out. Retry or read the saved state.'):errorMessage(e));
      }
    } finally {
      clearTimeout(timer);working.current=false;
      if(mounted.current){setBusy(false);onBusyChange(false);}
    }
  }
  if(hidden||!enabled||typeof bookmarked!=='boolean')return null;
  if(!owner)return <a className="post-bookmark-signin" href={signInHref(returnTo)}><Bookmark size={16}/>{t('登录收藏','Sign in to save')}</a>;
  return <div className="post-bookmark" aria-busy={busy} ref={root} onBlur={event=>{
    if(event.relatedTarget&&!event.currentTarget.contains(event.relatedTarget as Node))focusRetained.current=false;
  }}>
    <button ref={trigger} type="button" className={bookmarked?'active':''} aria-pressed={bookmarked}
      aria-label={bookmarked?t('取消收藏','Unsave post'):t('收藏帖子','Save post')}
      disabled={disabled||busy||uncertain} onClick={()=>void confirm(!bookmarked)}>
      <Bookmark size={16} fill={bookmarked?'currentColor':'none'}/>
      {busy?t('正在确认…','Confirming…'):bookmarked?t('已收藏','Saved'):t('收藏','Save')}
    </button>
    {error&&uncertain&&<div ref={recovery} className="post-bookmark-recovery">
      <SocialNotice error={error}/>
      <p>{t('收藏状态尚未确认。重试会保留原来的操作方向，也可以重新读取服务器状态。','The saved state is unconfirmed. Retry keeps the same intended action, or read the current server state.')}</p>
      {pending!==undefined&&<button type="button" disabled={disabled||busy} onClick={()=>void confirm(pending)}>
        {pending?t('重试收藏','Retry save'):t('重试取消收藏','Retry unsave')}
      </button>}
      <button type="button" disabled={disabled||busy} onClick={()=>void confirm()}>{t('重新读取状态','Read saved state')}</button>
    </div>}
  </div>;
}
