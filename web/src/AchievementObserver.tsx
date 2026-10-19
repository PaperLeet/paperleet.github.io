import AppLink from './AppLink';
import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { ArrowRight, X } from 'lucide-react';
import { achievementProgress, newlyUnlocked, type Achievement } from './achievementProgress';
import { AchievementMedal } from './AchievementMedal';
import { accountAchievementProgress } from './accountAchievements';
import { useAccountAchievements } from './useAccountAchievements';
import { useAccount } from './social';
import { useI18n } from './i18n';
import { hostedAppAvailable, hostedAppHref } from './deployment';
import { signInHref } from './accountNavigation';
import type { Paper, Problem, Saved } from './types';
import './achievements.css';

export default function AchievementObserver({problems,papers,saved,ready}: {problems:Problem[];papers:Paper[];saved:Record<string,Saved>;ready:boolean}) {
  const {t} = useI18n();
  const {user} = useAccount();
  const trigger = JSON.stringify(Object.entries(saved).filter(([,value]) => value.report || value.acceptedReport).map(([id,value]) => [id,value.report?.source_sha256,value.report?.summary,value.report?.error,value.acceptedReport?.source_sha256]));
  const account = useAccountAchievements(trigger);
  const previous = useRef<Set<string> | undefined>(undefined);
  const owner = user?.id || 'browser';
  const previousOwner = useRef(owner);
  const [celebration,setCelebration] = useState<Achievement[]>();
  const [hovered,setHovered] = useState(false);
  const [focused,setFocused] = useState(false);
  const available = !account.accountLoading && (user ? !!account.snapshot && problems.length > 0 : ready);
  const earned = available ? (user ? accountAchievementProgress(account.snapshot!,problems,papers) : achievementProgress(problems,papers,saved)).earned : [];
  const signature = JSON.stringify(earned.map(badge => badge.id));
  useEffect(() => {
    if (previousOwner.current !== owner) {previous.current = undefined;previousOwner.current = owner;setCelebration(undefined);}
    if (!available) { setCelebration(undefined); return; }
    const current = new Set(earned.map(badge => badge.id));
    // Existing progress is a baseline, not a new achievement on every visit.
    const unlocked = newlyUnlocked(previous.current,earned);
    if (unlocked.length) setCelebration(unlocked);
    previous.current = current;
  }, [available,signature,owner]);
  useEffect(() => {
    if (!celebration || hovered || focused) return;
    const timeout = setTimeout(() => setCelebration(undefined),12_000);
    return () => clearTimeout(timeout);
  }, [celebration,hovered,focused]);
  if (!available || previousOwner.current !== owner || !celebration?.length) return null;
  const badge = celebration[0];
  return <aside key={celebration.map(badge => badge.id).join(',')} className="achievement-celebration" data-tier={badge.tier} aria-label={t('新勋章','New achievement')} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)} onFocus={() => setFocused(true)} onBlur={event => { if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) setFocused(false); }}>
    <span className="achievement-celebration-rays" aria-hidden="true" /><span className="achievement-celebration-ring" aria-hidden="true" /><span className="achievement-celebration-particles" aria-hidden="true">{Array.from({length:8},(_,index) => <i key={index} style={{'--particle':index} as CSSProperties}/>)}</span>
    <AchievementMedal badge={badge}/><div className="achievement-celebration-copy"><span className="achievement-next-label">{user ? t('账号勋章已解锁','ACCOUNT ACHIEVEMENT UNLOCKED') : t('练习成就已解锁','ACHIEVEMENT UNLOCKED')}</span><p role="status" aria-live="polite" aria-atomic="true"><b>{t(...badge.title)}</b>{celebration.length > 1 && <small>{t(`同时点亮 ${celebration.length} 枚勋章`,`${celebration.length} achievements unlocked together`)}</small>}</p>
      {!user && hostedAppAvailable && <><small>{t('登录后提交解答，积累账号勋章。','Sign in and submit your solutions to earn account achievements.')}</small><AppLink href={hostedAppHref(signInHref('#/achievements'))} onClick={() => setCelebration(undefined)}>{t('免费注册 / 登录','Join free / Sign in')}<ArrowRight size={13} aria-hidden="true" /></AppLink><br/></>}
      <AppLink href="#/achievements" onClick={() => setCelebration(undefined)}>{user ? t('查看并置顶','View and feature') : t('查看勋章册','View your collection')}<ArrowRight size={13} aria-hidden="true" /></AppLink></div><button className="icon-button achievement-dismiss" type="button" aria-label={t('关闭勋章提示','Dismiss achievement notification')} onClick={() => setCelebration(undefined)}><X size={15} aria-hidden="true" /></button>
  </aside>;
}
