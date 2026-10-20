import {ArrowDown,ArrowUp,ArrowUpDown,ArrowDownWideNarrow,BookOpen,Bookmark,Building2,Code2,List,Rows3,X} from 'lucide-react';
import {useI18n} from './i18n';
import type {LibraryFilters,LibrarySort} from './libraryNavigation';
export type Props = {listId?:string;hasFilters:boolean; moduleView:boolean; company:string; course?:boolean; sort:LibrarySort; density:'comfortable'|'compact'; onClear:()=>void; onSort:(sort:LibrarySort)=>void; onDensity:(density:'comfortable'|'compact')=>void;onView?:(view:'papers'|'modules')=>void};
export default function LibraryResultControls({listId,hasFilters,moduleView,company,course,sort,density,onClear,onSort,onDensity,onView}:Props){
 const {t}=useI18n(),directModules=moduleView&&!company&&!course;
 return <div className="library-result-actions">{onView&&<div className="library-view-switch" role="group" aria-label={t('列表视图','List view')}>{(['papers','modules'] as const).map(view=><button type="button" key={view} aria-controls={listId} aria-pressed={(view==='modules')===moduleView} onClick={()=>{if((view==='modules')!==moduleView)onView(view);}}>{view==='papers'?<BookOpen size={16} aria-hidden="true"/>:<Code2 size={16} aria-hidden="true"/>}{view==='papers'?t('论文','Papers'):t('题目','Exercises')}</button>)}</div>}{hasFilters && <button type="button" className="text-button library-reset-filters" aria-controls={listId} title={t("清除筛选，保留当前视图、排序和每页数量","Clear filters; keep your view, order and page size")} onClick={onClear}><X size={16} aria-hidden="true"/>{t("重置筛选", "Reset filters")}</button>}<label className="library-sort" data-active={sort!=='default'||undefined}><ArrowDownWideNarrow size={16} aria-hidden="true" /><select aria-controls={listId} aria-label={directModules?t("题目排序","Sort exercises"):moduleView ? t("按来源排序", "Sort by source") : t("论文排序", "Sort papers")} value={sort} onChange={event => onSort(event.target.value as LibrarySort)}><option value="default">{t(company?"按分享者数":course?"课程顺序":"编号顺序", company?"Contributors":course ? "Course order" : "Number order")}</option><option value="newest">{directModules?t("来源最新","Newest source"):t("最新发表", "Newest first")}</option><option value="oldest">{directModules?t("来源最早","Oldest source"):t("最早发表", "Oldest first")}</option><option value="title">{t("名称 A–Z", "Title A–Z")}</option>{directModules&&<optgroup label={t('按练习安排','Plan your practice')}><option value="difficulty">{t('难度：简单优先','Difficulty: easy first')}</option><option value="difficulty-desc">{t('难度：困难优先','Difficulty: hard first')}</option><option value="duration">{t('用时：短优先','Time: short first')}</option><option value="duration-desc">{t('用时：长优先','Time: long first')}</option></optgroup>}</select></label><div className="library-density" role="group" aria-label={t('列表密度', 'List density')}><button type="button" aria-label={t('舒适列表', 'Comfortable list')} title={t('舒适列表：显示题目简介', 'Comfortable list: show descriptions')} aria-controls={listId} aria-pressed={density === 'comfortable'} onClick={() => onDensity('comfortable')}><Rows3 size={18} aria-hidden="true" /><span className="library-density-label">{t('舒适','Comfortable')}</span></button><button type="button" aria-label={t('紧凑列表', 'Compact list')} title={t('紧凑列表：隐藏题目简介', 'Compact list: hide descriptions')} aria-controls={listId} aria-pressed={density === 'compact'} onClick={() => onDensity('compact')}><List size={18} aria-hidden="true" /><span className="library-density-label">{t('紧凑','Compact')}</span></button></div></div>;
}

/** Retire hidden or stale filter actions before either focus or route mutation. */
function libraryFilterRoot(button:HTMLButtonElement,href:string){
 const root=button.closest<HTMLElement>('.paper-library');
 return root&&button.isConnected&&!button.closest('[hidden],[aria-hidden="true"],[inert]')&&window.location.href===href?root:undefined;
}

export function LibraryActiveFilters({filters,categoryLabel,onChange}:{filters:LibraryFilters;categoryLabel:string;onChange:(patch:Partial<LibraryFilters>)=>void}){
 const {t}=useI18n(),{query,category,tag,difficulty:diff,state}=filters;
 const href=typeof window==='undefined'?'':window.location.href;
 const items=[
  {id:'query',value:!!query,kind:['关键词','Keyword'],label:`“${query}”`,patch:{query:''}},
  {id:'category',value:category!=='all',kind:['分类','Category'],label:categoryLabel,patch:{category:'all',tag:'all'}},
  {id:'tag',value:tag!=='all',kind:['标签','Tag'],label:tag,patch:{tag:'all'}},
  {id:'difficulty',value:diff!=='all',kind:['难度','Difficulty'],label:({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[diff],patch:{difficulty:'all'}},
  {id:'state',value:state!=='all',kind:['状态','Status'],label:({todo:t('未开始','Not started'),started:t('进行中','In progress'),done:t('已完成','Solved')} as Record<string,string>)[state],patch:{state:'all'}},
 ].filter(item=>item.value);
 function remove(button:HTMLButtonElement,patch:Partial<LibraryFilters>){
  const root=libraryFilterRoot(button,href);if(!root)return;
  if(document.activeElement===button){
   const remaining=new Set(items.filter(item=>!(item.id in patch)).map(item=>item.id));
   const buttons=[...button.parentElement!.querySelectorAll<HTMLButtonElement>('button[data-filter-id]')],index=buttons.indexOf(button);
   const survives=(node:HTMLButtonElement)=>remaining.has(node.dataset.filterId||'');
   const target=buttons.slice(index+1).find(survives)||buttons.slice(0,index).reverse().find(survives)||root.querySelector<HTMLInputElement>('.library-toolbar .search input');
   target?.focus({preventScroll:true});
  }
  onChange(patch);
 }
 return <div className="active-filters" aria-label={t('当前筛选条件','Active filters')}>
  {items.map(item=>{const kind=t(item.kind[0],item.kind[1]);return <button type="button" key={item.id} data-filter-id={item.id} aria-label={t(`移除${kind}筛选：${item.label}`,`Remove ${kind.toLowerCase()} filter: ${item.label}`)} title={`${kind}: ${item.label}`} onClick={event=>remove(event.currentTarget,item.patch)}><span className="library-filter-kind">{kind}</span><span className="library-filter-value">{item.label}</span><X size={14} aria-hidden="true"/></button>;})}
 </div>;
}

export type LibraryHeaderProps={moduleView:boolean;company:string;course?:boolean;sort:LibrarySort;listId:string;onSort:(sort:LibrarySort)=>void};
const sortNames:Record<LibrarySort,[string,string]>={default:['编号顺序','Number order'],newest:['来源最新','Newest source first'],oldest:['来源最早','Oldest source first'],title:['名称 A–Z','Title A–Z'],difficulty:['简单优先','Easy first'],'difficulty-desc':['困难优先','Hard first'],duration:['用时短优先','Short time first'],'duration-desc':['用时长优先','Long time first']};
/** Optional column controls reuse the existing local ordering contract and native select. */
export function LibraryTableHeaders({moduleView,company,course,sort,listId,onSort}:LibraryHeaderProps){
 const {t}=useI18n(),direct=moduleView&&!company&&!course;
 function column(field:string,label:[string,string],orders:[LibrarySort,...LibrarySort[]]){
  const index=orders.indexOf(sort),active=index>=0,next=active?orders[index+1]||'default':orders[0],descending=sort==='newest'||sort.endsWith('-desc');
  const name=t(...label),current=active?t(...sortNames[sort]):t('未按此列排序','Not ordered by this column'),action=t(...sortNames[next]),description=t(`${name}：${current}。切换为${action}。`,`${name}: ${current}. Switch to ${action}.`),Icon=active?descending?ArrowDown:ArrowUp:ArrowUpDown;
  return <button key={field} type="button" className={'library-column-sort'+(field==='duration'||field==='year'?' paper-year':'')} data-sort-column={field} aria-pressed={active} aria-controls={listId} aria-label={description} title={description} onClick={()=>onSort(next)}><span className="library-column-label">{name}</span>{field==='difficulty'&&<span className="library-column-short" aria-hidden="true">{t('难度','Level')}</span>}<Icon size={13} aria-hidden="true"/></button>;
 }
 return <><span>{t('状态','Status')}</span>{!moduleView||direct?column('title',moduleView?['练习','Module']:['论文','Paper'],['title']):<span>{t('练习','Module')}</span>}{!moduleView?column('year',['年份','Year'],['newest','oldest']):direct?column('duration',['用时','Time'],['duration','duration-desc']):<span className="paper-year">{t('用时','Time')}</span>}{direct?column('difficulty',['难度','Difficulty'],['difficulty','difficulty-desc']):<span>{moduleView?t('难度','Difficulty'):t('已通过 / 模块','Solved / Modules')}</span>}<span className="library-company-column-marker" title={t('题目操作','Exercise actions')}><span className="library-favorite-column-marker"><Bookmark size={16} aria-hidden="true"/></span><span className="library-source-column-marker"><Building2 size={16} aria-hidden="true"/></span><span className="sr-only">{t('题目操作','Exercise actions')}</span></span></>;
}
