import type {RefObject} from 'react';
import {COMMUNITY_TAG_PAGE_SIZE} from './communityTagBrowsing';
import {useI18n} from './i18n';
import './community-tag-choices.css';

export default function CommunityTagChoiceControls({shown,total,more,expanded,listId,statusId,compactRef,onMore,onCompact}:{shown:number;total:number;more:number;expanded:boolean;listId:string;statusId:string;compactRef:RefObject<HTMLButtonElement|null>;onMore:()=>void;onCompact:()=>void}){
 const {t}=useI18n();
 return <div className="community-tag-choice-controls">
  <span id={statusId} role="status" aria-live="polite" aria-atomic="true">{t(`显示 ${shown} / ${total} 个匹配标签`,`Showing ${shown} of ${total} matching tags`)}</span>
  {total>COMMUNITY_TAG_PAGE_SIZE&&<div role="group" aria-label={t('标签列表显示','Tag list display')}>
   {more>0&&<button type="button" aria-controls={listId} aria-describedby={statusId} onClick={onMore}>{t(`再显示 ${more} 个标签`,`Show ${more} more tags`)}</button>}
   <button type="button" ref={compactRef} aria-controls={listId} aria-describedby={statusId} aria-disabled={!expanded} onClick={onCompact}>{t('收起标签列表','Show fewer tags')}</button>
  </div>}
 </div>;
}
