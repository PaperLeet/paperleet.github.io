import {ArrowDownWideNarrow,BookOpen,Code2,List,Rows3,X} from 'lucide-react';
import {useI18n} from './i18n';
import type {LibraryFilters,LibrarySort} from './libraryNavigation';
export type Props = {hasFilters:boolean; moduleView:boolean; company:string; course?:boolean; sort:LibrarySort; density:'comfortable'|'compact'; onClear:()=>void; onSort:(sort:LibrarySort)=>void; onDensity:(density:'comfortable'|'compact')=>void;onView?:(view:'papers'|'modules')=>void};
export default function LibraryResultControls({hasFilters,moduleView,company,course,sort,density,onClear,onSort,onDensity,onView}:Props){
 const {t}=useI18n(),directModules=moduleView&&!company&&!course;
 return <div className="library-result-actions">{onView&&<div className="library-view-switch" role="group" aria-label={t('列表视图','List view')}>{(['papers','modules'] as const).map(view=><button type="button" key={view} aria-pressed={(view==='modules')===moduleView} onClick={()=>{if((view==='modules')!==moduleView)onView(view);}}>{view==='papers'?<BookOpen size={16} aria-hidden="true"/>:<Code2 size={16} aria-hidden="true"/>}{view==='papers'?t('论文','Papers'):t('题目','Exercises')}</button>)}</div>}{hasFilters && <button type="button" className="text-button" onClick={onClear}>{t("重置筛选", "Reset filters")}</button>}<label className="library-sort" data-active={sort!=='default'||undefined}><ArrowDownWideNarrow size={16} aria-hidden="true" /><select aria-label={directModules?t("题目排序","Sort exercises"):moduleView ? t("按来源排序", "Sort by source") : t("论文排序", "Sort papers")} value={sort} onChange={event => onSort(event.target.value as LibrarySort)}><option value="default">{t(company?"按分享者数":course?"课程顺序":"编号顺序", company?"Contributors":course ? "Course order" : "Number order")}</option><option value="newest">{directModules?t("来源最新","Newest source"):t("最新发表", "Newest first")}</option><option value="oldest">{directModules?t("来源最早","Oldest source"):t("最早发表", "Oldest first")}</option><option value="title">{t("名称 A–Z", "Title A–Z")}</option>{directModules&&<optgroup label={t('按练习安排','Plan your practice')}><option value="difficulty">{t('难度：简单优先','Difficulty: easy first')}</option><option value="difficulty-desc">{t('难度：困难优先','Difficulty: hard first')}</option><option value="duration">{t('用时：短优先','Time: short first')}</option><option value="duration-desc">{t('用时：长优先','Time: long first')}</option></optgroup>}</select></label><div className="library-density" role="group" aria-label={t('列表密度', 'List density')}><button type="button" aria-label={t('舒适列表', 'Comfortable list')} title={t('舒适列表', 'Comfortable list')} aria-pressed={density === 'comfortable'} onClick={() => onDensity('comfortable')}><Rows3 size={18} aria-hidden="true" /></button><button type="button" aria-label={t('紧凑列表', 'Compact list')} title={t('紧凑列表', 'Compact list')} aria-pressed={density === 'compact'} onClick={() => onDensity('compact')}><List size={18} aria-hidden="true" /></button></div></div>;
}

export function LibraryActiveFilters({filters,categoryLabel,onChange}:{filters:LibraryFilters;categoryLabel:string;onChange:(patch:Partial<LibraryFilters>)=>void}){
 const {t}=useI18n(),{query,category,tag,difficulty:diff,state}=filters;
 return (
<div className="active-filters" aria-label={t("当前筛选条件", "Active filters")}>
        {[
          { id:"query", value:!!query, label:`“${query}”`, patch:{query:""} },
          { id:"category", value:category!=="all", label:categoryLabel, patch:{category:"all",tag:"all"} },
          { id:"tag", value:tag!=="all", label:tag, patch:{tag:"all"} },
          { id:"difficulty", value:diff!=="all", label:({E:t("简单","Easy"),M:t("中等","Medium"),H:t("困难","Hard")} as Record<string,string>)[diff], patch:{difficulty:"all"} },
          { id:"state", value:state!=="all", label:({todo:t("未开始","Not started"),started:t("进行中","In progress"),done:t("已完成","Solved")} as Record<string,string>)[state], patch:{state:"all"} },
        ].filter(item=>item.value).map(item=><button type="button" key={item.id} aria-label={t(`移除筛选：${item.label}`,`Remove filter: ${item.label}`)} onClick={()=>onChange(item.patch)}>{item.label}<X size={12} aria-hidden="true" /></button>)}
      </div>
 );
}
