import {CalendarDays} from 'lucide-react';
import {useI18n} from './i18n';
import type {CommunityFeedState} from './communityNavigation';
import './community-feed-period.css';

export default function CommunityFeedPeriod({days,available,pending,onChange}:{days:CommunityFeedState['days'];available:boolean;pending:boolean;onChange:(days:CommunityFeedState['days'])=>void}){
 const {t}=useI18n();
 if(!available&&!days)return null;
 return <label className="community-period"><CalendarDays size={15} aria-hidden="true"/><span className="sr-only">{t('帖子发布时间','Post publication period')}</span><select aria-label={t('帖子发布时间','Post publication period')} value={days||'all'} disabled={pending||!available} onChange={event=>onChange(event.target.value==='7'?'7':event.target.value==='30'?'30':undefined)}>
  <option value="all">{t('不限时间','All time')}</option><option value="7">{t('近 7 天发布','Past 7 days')}</option><option value="30">{t('近 30 天发布','Past 30 days')}</option>
 </select></label>;
}
