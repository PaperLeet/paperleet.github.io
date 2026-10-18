import {X} from 'lucide-react';
import {useI18n} from './i18n';
import {communityCategories} from './communityCategories';
import type {CommunityFeedState} from './communityNavigation';
import './community-feed-filters.css';

/** Applied filters stay removable during empty, failed and gated requests. */
export default function CommunityFeedFilters({filters,onChange,boards=communityCategories,tags=[]}:{
 filters:CommunityFeedState;
 boards?:readonly {id:string;zh:string;en:string}[]; tags?:readonly {id:string;zh:string;en:string}[];
 onChange:(change:Partial<CommunityFeedState>)=>void;
}){
 const {t}=useI18n();
 const board=boards.find(row=>row.id===filters.category);
 const chips:{key:string;label:string;change:Partial<CommunityFeedState>}[]=[];
 if(filters.category)chips.push({key:'category',label:board?t(board.zh,board.en):filters.category,change:{category:''}});
 if(filters.tag){const tag=tags.find(row=>row.id===filters.tag);chips.push({key:'tag',label:tag?t(tag.zh,tag.en):filters.tag,change:{tag:undefined}});}
 if(filters.following)chips.push({key:'following',label:t('已关注的人','Following'),change:{following:false}});
 if(filters.saved)chips.push({key:'saved',label:t('我的收藏','Saved'),change:{saved:false}});
 if(filters.search)chips.push({key:'search',label:t(`搜索：${filters.search}`,`Search: ${filters.search}`),change:{search:''}});
 if(filters.sort)chips.push({key:'sort',label:filters.sort==='top'?t('最多赞','Most liked'):t('最近活跃','Recently active'),change:{sort:undefined}});
 if(filters.days)chips.push({key:'days',label:filters.days==='7'?t('近 7 天发布','Published in the past 7 days'):t('近 30 天发布','Published in the past 30 days'),change:{days:undefined}});
 if(!chips.length)return null;
 return <div className="community-feed-filters" role="group" aria-label={t('当前筛选','Applied filters')}>
   {chips.map(chip=><button key={chip.key} type="button" className="community-feed-filter" aria-label={t(`移除筛选：${chip.label}`,`Remove filter: ${chip.label}`)} onClick={()=>onChange(chip.change)}><span>{chip.label}</span><X size={14} aria-hidden="true"/></button>)}
   <button type="button" className="community-feed-reset" onClick={()=>onChange({...(filters.tag?{tag:undefined}:{}),category:'',following:false,saved:false,search:'',sort:undefined,...(filters.days?{days:undefined}:{})})}>{t('清除筛选','Clear filters')}</button>
 </div>;
}
