import { useEffect, useId, useRef, useState } from 'react';
import { Award, Bookmark, ChevronRight, Download, Globe2, Moon, Settings, ShieldCheck, Sun, UserRound } from 'lucide-react';
import { Avatar, profileHref, useAccount, type AccountUser } from './social';
import { useI18n } from './i18n';
import { hostedAppAvailable, hostedAppHref } from './deployment';
import { signInHref } from './accountNavigation';
import { deferredFeature } from './deferredFeature';

const AccountLogout=deferredFeature(()=>import('./AccountLogout'),['退出登录','sign out'],{loading:()=>null,failure:(_props,retry)=><LogoutRetry retry={retry}/>});
function LogoutRetry({retry}:{retry:()=>void}) {
  const {t}=useI18n();
  return <button type="button" className="account-menu-row" onClick={retry}>{t('重试加载退出登录','Retry loading sign out')}</button>;
}
export type HeaderAccountMenuProps={
  returnTo?:string;
  theme?:'light'|'dark';
  onTheme?:(theme:'light'|'dark')=>void;
  onBackup?:()=>void;
  backupReady?:boolean;
};

export default function HeaderAccountMenu(props:HeaderAccountMenuProps) {
  const {user,loading,capabilities}=useAccount();
  const {t}=useI18n();
  const [open,setOpen]=useState(false);
  const root=useRef<HTMLDivElement>(null);
  const trigger=useRef<HTMLButtonElement>(null);
  const panel=useRef<HTMLElement>(null);
  const id=useId();
  const close=()=>setOpen(false);
  useEffect(()=>{
    if(!open)return;
    const frame=requestAnimationFrame(()=>panel.current?.focus({preventScroll:true}));
    const outside=(event:PointerEvent)=>{if(!root.current?.contains(event.target as Node))setOpen(false);};
    const escape=(event:KeyboardEvent)=>{
      if(event.key!=='Escape'||event.defaultPrevented)return;
      event.preventDefault();setOpen(false);trigger.current?.focus({preventScroll:true});
    };
    window.addEventListener('hashchange',close);
    document.addEventListener('pointerdown',outside);
    document.addEventListener('keydown',escape);
    return()=>{cancelAnimationFrame(frame);window.removeEventListener('hashchange',close);document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);};
  },[open]);
  useEffect(()=>setOpen(false),[user?.id]);
  return <div className="account-menu" ref={root} onBlur={event=>{
    if(event.relatedTarget&&!event.currentTarget.contains(event.relatedTarget as Node))setOpen(false);
  }}>
    <button type="button" className="account-menu-trigger" ref={trigger} aria-haspopup="dialog" aria-expanded={open} aria-controls={id} aria-label={t('打开个人菜单','Open personal menu')} title={t('个人菜单','Personal menu')} onClick={()=>setOpen(value=>!value)} onKeyDown={event=>{
      if(event.key==='ArrowDown'){event.preventDefault();setOpen(true);}
    }}>{user?<Avatar user={user} size={32}/>:<span className="guest-avatar" aria-hidden="true"><UserRound size={18}/></span>}</button>
    {open&&<section className="account-menu-panel" ref={panel} tabIndex={-1} id={id} role="dialog" aria-label={t('个人菜单','Personal menu')}>
      <AccountMenuContent {...props} user={user} loading={loading} savedAvailable={capabilities?.community_bookmarks===true} onClose={()=>{close();trigger.current?.focus({preventScroll:true});}}/>
    </section>}
  </div>;
}

export function AccountMenuContent({user,loading,onClose,returnTo='',theme='light',onTheme,onBackup,backupReady,savedAvailable=false}:HeaderAccountMenuProps&{user:AccountUser|null;loading:boolean;onClose:()=>void;savedAvailable?:boolean}) {
  const {locale,setLocale,t}=useI18n();
  const identity=<><span className="account-menu-identity-avatar">{user?<Avatar user={user} size={46}/>:<UserRound size={22}/>}</span><span><b>{user?user.display_name:t('访客工作区','Guest workspace')}</b><small>{user?`@${user.username}`:t('偏好和游客草稿自动保存在当前浏览器','Preferences and guest drafts save in this browser')}</small></span>{user&&<ChevronRight size={16}/>}</>;
  return <>
    {user?<a className="account-menu-identity" href={hostedAppHref(profileHref(user))} onClick={onClose}>{identity}</a>:<div className="account-menu-identity">{identity}</div>}
    {!user&&hostedAppAvailable&&<a className="account-menu-signin button" href={hostedAppHref(signInHref(returnTo))} onClick={onClose}>{loading?t('正在检查登录状态…','Checking sign-in…'):t('登录账号','Sign in')}<ChevronRight size={14}/></a>}
    <div className="account-menu-shortcuts">
      {user?<a href={hostedAppHref(profileHref(user))} onClick={onClose}><UserRound size={21}/><span>{t('个人主页','My profile')}</span></a>:<button type="button" disabled={!backupReady||!onBackup} onClick={()=>{onClose();onBackup?.();}}><Download size={21}/><span>{t('代码备份','Code backups')}</span></button>}
      <a href="#/achievements" onClick={onClose}><Award size={22}/><span>{t('学习勋章','Achievements')}</span></a>
      <a href="#/settings" onClick={onClose}><Settings size={21}/><span>{t('设置','Settings')}</span></a>
    </div>
    <div className="account-menu-preferences">
      <div className="account-menu-preference"><span><Globe2 size={16}/>{t('语言','Language')}</span><div className="account-menu-segment" role="group" aria-label={t('界面语言','Interface language')}><button type="button" lang="zh-CN" aria-pressed={locale==='zh'} onClick={()=>setLocale('zh')}>中文</button><button type="button" lang="en" aria-pressed={locale==='en'} onClick={()=>setLocale('en')}>EN</button></div></div>
      <div className="account-menu-preference"><span><Sun size={16}/>{t('外观','Appearance')}</span><div className="account-menu-segment" role="group" aria-label={t('外观','Appearance')}><button type="button" aria-pressed={theme==='light'} disabled={!onTheme} onClick={()=>onTheme?.('light')}><Sun size={14}/>{t('浅色','Light')}</button><button type="button" aria-pressed={theme==='dark'} disabled={!onTheme} onClick={()=>onTheme?.('dark')}><Moon size={14}/>{t('深色','Dark')}</button></div></div>
    </div>
    <div className="account-menu-links">
      <a className="account-menu-row" href="#/favorites" onClick={onClose}><Bookmark size={16}/><span>{t("收藏题目","Saved problems")}</span><ChevronRight size={14}/></a>
      {user&&savedAvailable&&<a className="account-menu-row" href={hostedAppHref('#/interviews?saved=1&days=all')} onClick={onClose}><Bookmark size={16}/><span>{t('收藏面经','Saved interviews')}</span><ChevronRight size={14}/></a>}
      {user&&<a className="account-menu-row" href={hostedAppHref('#/account')} onClick={onClose}><ShieldCheck size={16}/><span>{t('账号与隐私','Account & privacy')}</span><ChevronRight size={14}/></a>}
      {user&&onBackup&&<button type="button" className="account-menu-row" disabled={!backupReady} onClick={()=>{onClose();onBackup();}}><Download size={16}/><span>{t('代码备份','Code backups')}</span><ChevronRight size={14}/></button>}
    </div>
    {user?<AccountLogout/>:<p className="account-menu-footnote">{t('开始练习无需登录','You can practice without signing in')}</p>}
  </>;
}
