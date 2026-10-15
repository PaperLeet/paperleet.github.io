import { useEffect, useState } from 'react';
import { ArrowRight, Award, LockKeyhole, Pin } from 'lucide-react';
import { useI18n } from './i18n';
import { achievementDefinitions } from './achievementProgress';
import { AchievementMedal } from './AchievementMedal';
import { readPublicAchievements, type PublicAchievements } from './accountAchievements';
import { DateLabel, socialApi, useAccount } from './social';
import './achievements.css';

export default function ProfileAchievements({username,userId,isSelf,isPublic}: {username:string;userId:string;isSelf:boolean;isPublic:boolean}) {
  const {capabilities,loading} = useAccount();
  const supported = !!capabilities?.account_achievements;
  const [data,setData] = useState<PublicAchievements>();
  const [error,setError] = useState(false);
  const [revision,setRevision] = useState(0);
  useEffect(() => {
    if (!supported || loading) return;
    const controller = new AbortController();
    setError(false);
    socialApi<unknown>(`community/profiles/${encodeURIComponent(username)}/achievements`,undefined,controller.signal)
      .then(value => { if (!controller.signal.aborted) setData(readPublicAchievements(value,userId)); })
      .catch(() => { if (!controller.signal.aborted) {setData(undefined);setError(true);} });
    return () => controller.abort();
  },[username,userId,supported,loading,revision]);
  useEffect(() => {
    if (!supported) return;
    const refresh = () => { if (document.visibilityState === 'visible') setRevision(value => value + 1); };
    const changed = (event:Event) => { if ((event as CustomEvent).detail === userId) refresh(); };
    window.addEventListener('focus',refresh);
    window.addEventListener('paperleet-achievements-changed',changed);
    const timer = window.setInterval(refresh,60_000);
    return () => {window.removeEventListener('focus',refresh);window.removeEventListener('paperleet-achievements-changed',changed);window.clearInterval(timer);};
  },[supported,userId]);
  if (!supported || loading) return null;
  return <ProfileAchievementShowcase data={data} error={error} isSelf={isSelf} isPublic={isPublic} onRetry={() => setRevision(value => value + 1)}/>;
}

export function ProfileAchievementShowcase({data,error,isSelf,isPublic,onRetry}: {data?:PublicAchievements;error:boolean;isSelf:boolean;isPublic:boolean;onRetry:()=>void}) {
  const {t} = useI18n();
  const earned = data?.visible ? data.earned : [];
  const pinned = data?.visible ? data.pinned_ids.flatMap(id => {
    const badge = achievementDefinitions.find(badge => badge.id === id);
    const award = earned.find(award => award.id === id);
    return badge && award ? [{...badge,...award,unlocked:true}] : [];
  }) : [];
  return <section className="social-panel profile-achievements" aria-labelledby="profile-achievements-title" aria-busy={!data && !error}>
    <header><div><h2 id="profile-achievements-title"><Award size={18} aria-hidden="true" />{t('学习勋章','Achievements')}{data?.visible && <span>{earned.length}</span>}</h2></div>{isSelf && <a href="#/achievements">{t('管理置顶','Manage featured')}<ArrowRight size={13} aria-hidden="true" /></a>}</header>
    {error ? <p className="profile-achievement-empty" role="alert">{t('暂时无法加载勋章。','Could not load achievements.')} <button type="button" className="text-button" onClick={onRetry}>{t('重试','Retry')}</button></p> : !data ? <p className="profile-achievement-empty" role="status">{t('正在同步勋章…','Syncing achievements…')}</p> : !data.visible ? <p className="profile-achievement-empty"><LockKeyhole size={15} aria-hidden="true" />{t('学习勋章仅本人可见。','Learning achievements are private.')}</p> : <>
      {isSelf && !isPublic && <p className="profile-achievement-private"><LockKeyhole size={12} aria-hidden="true" />{t('仅自己可见 · 跟随学习进度隐私设置','Only visible to you · follows learning progress privacy')}</p>}
      {pinned.length ? <ol className="profile-featured-medals" aria-label={t('置顶勋章','Featured achievements')}>{pinned.map(badge => <li key={badge.id} data-tier={badge.tier}><AchievementMedal badge={badge}/><div><span><Pin size={10} aria-hidden="true" />{t('置顶勋章','FEATURED')}</span><h3>{t(...badge.title)}</h3><p>{t(...badge.description)}</p><small>{t('解锁于','Unlocked')} <DateLabel value={badge.earned_at}/></small></div></li>)}</ol> : <p className="profile-achievement-empty">{earned.length ? (isSelf ? t('还没有置顶勋章，挑几枚代表你的研究旅程。','Choose a few achievements to represent your research journey.') : t('尚未设置置顶勋章。','No featured achievements yet.')) : (isSelf ? t('每一次完整通过，都会让第一枚勋章更近一点。','Every complete pass brings your first achievement closer.') : t('新的研究旅程，正等待第一枚勋章。','A new research journey awaits its first achievement.'))}</p>}
      {!!earned.length && <details className="profile-achievement-collection" open={!pinned.length}><summary>{t(`查看全部 ${earned.length} 枚勋章`,`View all ${earned.length} achievements`)}</summary><div>{earned.map(award => {
        const badge = achievementDefinitions.find(badge => badge.id === award.id)!;
        return <article key={badge.id}><AchievementMedal badge={{...badge,unlocked:true}}/><div><h3>{t(...badge.title)}</h3><p>{t(...badge.description)}</p><small><DateLabel value={award.earned_at}/></small></div></article>;
      })}</div></details>}
    </>}
  </section>;
}
