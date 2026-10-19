import AppLink from './AppLink';
import { useState } from 'react';
import type { ReactNode } from 'react';
import { ArrowLeft, ArrowRight, Award, Check, Circle, Cloud, LockKeyhole } from 'lucide-react';
import { useI18n } from './i18n';
import { achievementProgress } from './achievementProgress';
import { AchievementMedal } from './AchievementMedal';
import AchievementPins from './AchievementPins';
import { accountAchievementProgress } from './accountAchievements';
import { useAccountAchievements } from './useAccountAchievements';
import { DateLabel, profileHref, useAccount, type AccountUser } from './social';
import { signInHref } from './accountNavigation';
import { hostedAppAvailable, hostedAppHref } from './deployment';
import type { Paper, Problem, Saved } from './types';
import './achievements.css';

export type AchievementsProps = {problems:Problem[];papers:Paper[];saved:Record<string,Saved>;ready:boolean;returnTo:string;workspaceName:string;onRetry:()=>void;error?:string};
export default function Achievements(props:AchievementsProps) {
  const {user,loading} = useAccount();
  return <AchievementCollection key={loading ? 'pending' : user?.id || 'browser'} {...props}/>;
}
function AchievementCollection({problems,papers,saved,ready,returnTo,workspaceName,onRetry,error}: AchievementsProps) {
  const {t} = useI18n();
  const {user} = useAccount();
  const account = useAccountAchievements('collection');
  const progress = account.accountLoading ? undefined : user ? (account.snapshot && problems.length ? accountAchievementProgress(account.snapshot,problems,papers) : undefined) : ready ? achievementProgress(problems,papers,saved) : undefined;
  const unavailable = user && !account.accountLoading && !account.supported ? t('当前服务尚未支持账号勋章同步，请更新账号服务后重试。','This service does not support account achievement sync yet. Update the account service and try again.') : '';
  const retry = () => {account.refresh();onRetry();};
  return <AchievementCollectionView progress={progress} returnTo={returnTo} workspaceName={workspaceName} error={account.error || unavailable || error} onRetry={retry} user={user} syncError={account.error} onSyncRetry={account.refresh} featured={user && account.snapshot && progress ? <AchievementPins snapshot={account.snapshot} badges={progress.badges} isPublic={user.show_progress} onSaved={account.accept} onRefresh={account.refresh} disabled={!!account.error}/> : undefined}/>;
}

export function AchievementCollectionView({progress,returnTo,workspaceName,error:loadError,onRetry:retry,user,syncError,onSyncRetry,featured}: {
  progress?:ReturnType<typeof achievementProgress>;returnTo:string;workspaceName:string;error?:string;onRetry:()=>void;
  user?:AccountUser|null;syncError?:string;onSyncRetry?:()=>void;featured?:ReactNode;
}) {
  const {t} = useI18n();
  const [filter,setFilter] = useState<'all'|'earned'|'locked'>('all');
  const visible = progress?.badges.filter(badge => filter === 'all' || badge.unlocked === (filter === 'earned')) || [];
  return <main className="achievements-page">
    <AppLink className="back-link achievement-back" href={returnTo}><ArrowLeft size={15} aria-hidden="true" />{t('返回刚才的页面','Back to where you left off')}</AppLink>
    <header className="achievement-heading"><h1>{t('学习勋章','Learning achievements')}</h1></header>
    {!progress ? <section className="achievement-load" aria-busy={!loadError}>{loadError ? <><p role="alert">{loadError}</p><button className="button subtle" onClick={retry}>{t('重试加载','Try again')}</button></> : <><span className="spinner"/><p role="status">{user ? t('正在同步账号勋章…','Syncing account achievements…') : t('正在核对工作区的通过记录…','Checking your workspace’s completed modules…')}</p></>}</section> : <>
      {user && <div className="achievement-sync-status"><span><Cloud size={15} aria-hidden="true" />{t('已连接账号勋章','Account achievements connected')}</span><AppLink href={profileHref(user)}>{t('查看个人主页','View your profile')}<ArrowRight size={13} aria-hidden="true" /></AppLink></div>}
      {syncError && <p className="achievement-sync-error" role="alert">{syncError} <button type="button" className="text-button" onClick={onSyncRetry}>{t('重试同步','Retry sync')}</button></p>}
      <section className="achievement-hero" aria-label={t('勋章概览','Achievement overview')}>
        <div className="achievement-totals"><span className="achievement-workspace">{workspaceName}</span><div className="achievement-count"><b>{progress.earned.length}</b><span>/ {progress.badges.length}<small>{t('枚勋章已点亮','achievements unlocked')}</small></span></div><div className="achievement-total-track" role="progressbar" aria-label={t('勋章收集进度','Achievement collection progress')} aria-valuenow={progress.earned.length} aria-valuemin={0} aria-valuemax={progress.badges.length}><span style={{width:`${progress.earned.length / progress.badges.length * 100}%`}}/></div><div className="achievement-proof-stats"><span><Check size={13} aria-hidden="true" />{progress.solved} {t('个模块通过','modules passed')}</span><span>{progress.completedPapers} {t('篇论文完成','papers completed')}</span></div></div>
        {progress.next ? <div className="achievement-next"><AchievementMedal badge={{...progress.next,unlocked:true}} large/><div><span className="achievement-next-label">{t('下一个目标','YOUR NEXT MILESTONE')}</span><h2>{t(...progress.next.title)}</h2><p>{t(...progress.next.description)}</p><AppLink href={progress.next.href}>{t('继续点亮','Keep going')}<ArrowRight size={14} aria-hidden="true" /></AppLink></div></div> : <div className="achievement-next"><AchievementMedal badge={progress.earned[progress.earned.length - 1]} large/><div><span className="achievement-next-label">{t('全部点亮','ALL UNLOCKED')}</span><h2>{t('让理解继续生长','Keep your curiosity growing')}</h2><p>{t('回看实验与笔记，把收获带到下一个想法里。','Revisit the experiments and take what you learned into your next idea.')}</p></div></div>}
      </section>
      {featured}
      {!user && hostedAppAvailable && <section className="achievement-account-prompt"><div><h2><Cloud size={16} aria-hidden="true" />{t('建立你的学习档案','Build your learning profile')}</h2><p>{t('免费注册后，提交解答并通过测试即可积累账号勋章，还能选 3 枚展示在个人主页。当前练习记录保留在此设备，不会自动上传。','Join free, then submit passing solutions to earn account achievements and feature up to 3 on your profile. Your current practice records stay on this device and are not uploaded automatically.')}</p></div><AppLink className="button" href={hostedAppHref(signInHref('#/achievements'))}>{t('免费注册 / 登录','Join free / Sign in')}<ArrowRight size={13} aria-hidden="true" /></AppLink></section>}
      <div className="achievement-library-heading"><h2><Award size={17} aria-hidden="true" />{t('我的勋章册','Your collection')}</h2><div className="achievement-filters" role="group" aria-label={t('筛选勋章','Filter achievements')}>{(['all','earned','locked'] as const).map(value => <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>{value === 'all' ? t('全部','All') : value === 'earned' ? t('已解锁','Unlocked') : t('待解锁','Locked')}<span>{value === 'all' ? progress.badges.length : value === 'earned' ? progress.earned.length : progress.badges.length - progress.earned.length}</span></button>)}</div></div>
      <div className="achievement-grid">
        {visible.map(badge => <article className="achievement-card" key={badge.id} data-earned={badge.unlocked} data-tier={badge.tier}>
          <div className="achievement-card-top"><span className="achievement-tier">{({bronze:t('启程','BEGINNINGS'),silver:t('进阶','EXPLORATION'),gold:t('卓越','MASTERY'),prism:t('典藏','DISTINCTION')})[badge.tier]}</span>{badge.unlocked ? <span className="achievement-status earned"><Check size={12} aria-hidden="true" />{t('已解锁','Unlocked')}</span> : <span className="achievement-status"><LockKeyhole size={11} aria-hidden="true" />{t('待解锁','Locked')}</span>}</div>
          <AchievementMedal badge={badge}/><h3>{t(...badge.title)}</h3><p>{t(...badge.description)}</p>
          <div className="achievement-card-bottom"><label><span>{badge.unlocked ? t('目标已达成','Milestone reached') : t('当前进度','Your progress')}</span><b>{badge.unlocked ? badge.target : Math.min(badge.current,badge.target)} / {badge.target}</b><progress aria-label={t(...badge.title)} max={badge.target} value={badge.unlocked ? badge.target : Math.min(badge.current,badge.target)} /></label>{badge.unlocked ? <span className="achievement-complete"><Circle size={5} fill="currentColor" aria-hidden="true" />{badge.earnedAt ? <>{t('解锁于','Unlocked')} <DateLabel value={badge.earnedAt}/></> : t('每一步，都算数','Every step counts')}</span> : <AppLink href={badge.href}>{t('去练习','Practice toward this')}<ArrowRight size={13} aria-hidden="true" /></AppLink>}</div>
        </article>)}
      </div>
      {!visible.length && <div className="achievement-empty"><Award size={32} aria-hidden="true" /><h3>{filter === 'earned' ? t('第一枚勋章，在下一次通过之后','Your first achievement starts with a pass') : t('这一页已经全部点亮','You’ve unlocked them all')}</h3><p>{t('每枚勋章的解锁条件都写在勋章卡片上。','Each card shows exactly what it takes to unlock it.')}</p><button className="button subtle" onClick={() => setFilter('all')}>{t('查看全部勋章','See all achievements')}</button></div>}
      <p className="achievement-source-note">{user ? t('账号勋章根据正式提交的完整测试通过记录同步，重复通过只计一次，已解锁的勋章会保留。练习通过后请提交解答，才能获得账号勋章。公开展示遵循你的学习进度隐私设置。','Account achievements use complete-suite passes from saved submissions. Repeated passes count once and earned achievements are retained. Submit your passing solutions to earn account achievements. Public display follows your learning progress privacy setting.') : t('这里展示当前工作区的练习成就，同一道题重复通过只计一次。这些记录保存在此设备；登录后提交解答，才能积累账号勋章。','These practice achievements use complete-suite passes in your current workspace. Repeated passes count once. These records stay on this device; sign in and submit your solutions to earn account achievements.')}</p>
    </>}
  </main>;
}
