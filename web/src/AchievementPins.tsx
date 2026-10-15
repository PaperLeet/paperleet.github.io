import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, Check, Pin, X } from 'lucide-react';
import { useI18n } from './i18n';
import { socialApi, SocialError } from './social';
import { AchievementMedal } from './AchievementMedal';
import type { Achievement } from './achievementProgress';
import { movePinned, readAccountAchievements, type AccountAchievements } from './accountAchievements';
import { announceAchievementChange } from './useAccountAchievements';

export default function AchievementPins({snapshot,badges,isPublic,onSaved,onRefresh,disabled}: {
  snapshot:AccountAchievements; badges:Achievement[]; isPublic:boolean; disabled:boolean;
  onSaved:(snapshot:AccountAchievements)=>void; onRefresh:()=>void;
}) {
  const {t} = useI18n();
  const [draft,setDraft] = useState<{ids:string[];revision:number}>();
  const [busy,setBusy] = useState(false);
  const [message,setMessage] = useState('');
  const [failed,setFailed] = useState(false);
  const request = useRef<AbortController | undefined>(undefined);
  useEffect(() => () => request.current?.abort(),[]);
  const ids = draft?.ids || snapshot.pinned_ids;
  const changed = JSON.stringify(ids) !== JSON.stringify(snapshot.pinned_ids);
  const conflict = !!draft && draft.revision !== snapshot.revision;
  const blocked = busy || disabled || conflict;
  function edit(next:string[]) {
    setDraft({ids:next,revision:draft?.revision ?? snapshot.revision});
    setMessage(''); setFailed(false);
  }
  async function save() {
    if (!draft || !changed || blocked) return;
    const controller = new AbortController(); request.current = controller;
    setBusy(true); setMessage(''); setFailed(false);
    try {
      const data = await socialApi<unknown>('account/achievements/pins',{
        expected_user_id:snapshot.user_id,expected_revision:draft.revision,pinned_ids:ids,
      },controller.signal);
      const next = readAccountAchievements(data,snapshot.user_id);
      if (controller.signal.aborted) return;
      onSaved(next); setDraft(undefined);
      setMessage(t('置顶已保存，其他设备和个人主页会同步更新。','Pins saved. Your profile and other devices will update.'));
      announceAchievementChange(snapshot.user_id);
    } catch (error) {
      if (controller.signal.aborted) return;
      setFailed(true);
      if (error instanceof SocialError && error.code === 'pins_changed') {
        setMessage(t('其他设备刚更新了置顶，请加载最新设置后重新选择。','Another device updated your pins. Load the latest settings and choose again.'));
        onRefresh();
      } else setMessage(t('保存失败，选择仍保留，请重试。','Could not save. Your selection is preserved; please retry.'));
    } finally { if (!controller.signal.aborted) setBusy(false); }
  }
  return <section className="achievement-pins" aria-labelledby="achievement-pins-title" aria-busy={busy}>
    <div className="achievement-pins-heading"><div><h2 id="achievement-pins-title"><Pin size={16} aria-hidden="true" />{t('个人主页 · 置顶勋章','Profile · featured achievements')}</h2><p>{t('选择最多 3 枚已解锁勋章，按你喜欢的顺序展示。','Choose up to 3 unlocked achievements and arrange their display order.')}</p></div><span className="achievement-pin-count">{ids.length} / 3</span></div>
    <ol className="achievement-pin-slots" aria-label={t('置顶展示顺序','Featured display order')}>
      {[0,1,2].map(index => {
        const badge = badges.find(badge => badge.id === ids[index]);
        return <li key={index} data-filled={!!badge}>{badge ? <><AchievementMedal badge={badge}/><div><b>{t(...badge.title)}</b><span>{t(`第 ${index + 1} 位`,`Position ${index + 1}`)}</span></div><div className="achievement-pin-controls"><button type="button" className="icon-button" disabled={blocked || index === 0} aria-label={t(`前移${badge.title[0]}`,`Move ${badge.title[1]} earlier`)} onClick={() => edit(movePinned(ids,badge.id,-1))}><ArrowLeft size={13} aria-hidden="true" /></button><button type="button" className="icon-button" disabled={blocked || index === ids.length - 1} aria-label={t(`后移${badge.title[0]}`,`Move ${badge.title[1]} later`)} onClick={() => edit(movePinned(ids,badge.id,1))}><ArrowRight size={13} aria-hidden="true" /></button><button type="button" className="icon-button" disabled={blocked} aria-label={t(`取消置顶${badge.title[0]}`,`Unpin ${badge.title[1]}`)} onClick={() => edit(ids.filter(id => id !== badge.id))}><X size={13} aria-hidden="true" /></button></div></> : <><span className="achievement-empty-pin"><Pin size={20} aria-hidden="true" /></span><div><b>{t('留给下一次突破','A place for your next milestone')}</b><span>{t(`置顶位置 ${index + 1}`,`Featured slot ${index + 1}`)}</span></div></>}</li>;
      })}
    </ol>
    <div className="achievement-pin-choices" role="group" aria-label={t('选择置顶勋章','Choose featured achievements')}>
      {badges.filter(badge => badge.unlocked).map(badge => <button key={badge.id} type="button" aria-pressed={ids.includes(badge.id)} disabled={blocked || (!ids.includes(badge.id) && ids.length >= 3)} onClick={() => edit(ids.includes(badge.id) ? ids.filter(id => id !== badge.id) : [...ids,badge.id])}>{ids.includes(badge.id) ? <Check size={13} aria-hidden="true" /> : <Pin size={12} aria-hidden="true" />}{t(...badge.title)}</button>)}
      {!badges.some(badge => badge.unlocked) && <p>{t('解锁第一枚账号勋章后，就可以把它放在这里。','Your first account achievement can take a place here once unlocked.')}</p>}
    </div>
    <div className="achievement-pin-footer"><p>{isPublic ? t('已保存的勋章会展示在个人主页。','Saved achievements appear on your profile.') : <>{t('学习进度目前仅自己可见，置顶勋章也会隐藏。','Your learning progress and featured achievements are currently private.')} <a href="#/account">{t('管理可见性','Manage visibility')}</a></>}</p><div>{conflict ? <button type="button" className="button subtle" onClick={() => {setDraft(undefined);setMessage('');setFailed(false);onRefresh();}}>{t('加载最新置顶','Load latest pins')}</button> : <><button type="button" className="text-button" disabled={!changed || busy} onClick={() => {setDraft(undefined);setMessage('');setFailed(false);}}>{t('取消更改','Cancel changes')}</button><button type="button" className="button" disabled={!changed || blocked} onClick={() => void save()}>{busy ? t('保存中…','Saving…') : t('保存置顶','Save pins')}</button></>}</div></div>
    {message && <p className="achievement-pin-message" role={failed ? 'alert' : 'status'}>{message}</p>}
    {conflict && !message && <p className="achievement-pin-message" role="alert">{t('置顶已在其他设备变更，请加载最新设置。','Your pins changed on another device. Load the latest settings.')}</p>}
  </section>;
}
