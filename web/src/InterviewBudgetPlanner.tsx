import {useId,useState} from 'react';
import {ChevronDown,Clock3,Plus} from 'lucide-react';
import {useI18n} from './i18n';
import {readPracticeBudget,type BudgetMode,type BudgetPlan} from './interviewBudgetPlan';
import type {InterviewPracticeGroupState} from './InterviewPracticeGroup';
import './interview-budget-planner.css';

export default function InterviewBudgetPlanner({group}:{group:InterviewPracticeGroupState}){
 const {t}=useI18n(),[open,setOpen]=useState(false),[raw,setRaw]=useState('60'),[mode,setMode]=useState<BudgetMode>(group.planningDefault);
 return <details className="interview-budget-planner" onToggle={event=>setOpen(event.currentTarget.open)}><summary><Clock3 size={16} aria-hidden="true"/><span>{t('按可用时间组题','Plan for your available time')}</span><ChevronDown size={15} aria-hidden="true"/></summary>{open&&<Budget group={group} raw={raw} setRaw={setRaw} mode={mode} setMode={setMode}/>}</details>;
}
function Budget({group,raw,setRaw,mode,setMode}:{group:InterviewPracticeGroupState;raw:string;setRaw:(value:string)=>void;mode:BudgetMode;setMode:(value:BudgetMode)=>void}){
 const {t,locale}=useI18n(),id=useId();
 const budget=readPracticeBudget(raw),plan=group.plan(budget,mode),invalid=budget===undefined;
 const title=(number:number)=>{const problem=group.byNumber.get(number);return problem?locale==='en'?problem.translations?.en?.title||problem.title:problem.title:String(number);};
 const reason=(plan:BudgetPlan)=>plan.reason==='budget'?t('请输入 1–240 的整数分钟。','Enter a whole number of minutes from 1 to 240.')
  :plan.reason==='pending'?group.pendingReason==='catalog'?t('先载入完整题库，再按当前范围安排。','Load the complete catalog before planning this view.'):group.pendingReason==='sources'?t('先读取当前来源范围，再安排练习。','Load the current source scope before planning.'):t('所选学习状态仍在等待进度，条件会保留。','The selected learning-state filter is waiting for progress.')
  :plan.reason==='unavailable'?t('已选组中有练习暂不可用，请先恢复题库或调整已选项。','Some selected modules are unavailable. Restore the catalog or adjust your selection first.')
  :plan.reason==='estimate'?t(`${plan.unknownSelected} 道已选练习没有可确认的估时，暂时不能核算时间预算。`,`${plan.unknownSelected} selected ${plan.unknownSelected===1?'exercise has':'exercises have'} no confirmed estimate, so the time budget cannot be checked.`)
  :plan.reason==='over'?t(`已选练习约需 ${plan.selectedMinutes} 分钟，超过预算 ${plan.selectedMinutes-plan.budget!} 分钟；已选项保持不变。`,`Your selection is about ${plan.selectedMinutes} min, ${plan.selectedMinutes-plan.budget!} min over the budget. Your choices are kept.`)
  :plan.reason==='full'?t('练习组已有 6 道，可以先调整已选项。','Your batch already has six exercises. Adjust your choices first.')
  :plan.reason==='progress'?t('进度尚未就绪。可以重试进度，或选择“全部练习”先安排。','Progress is not ready. Retry progress, or choose all exercises to plan without it.')
  :plan.reason==='empty'?t('当前范围没有可补充的练习。可调整列表筛选或安排已通过题目复习。','No additional exercises match this scope. Adjust the list filters or include passed exercises for review.')
  :plan.reason==='fit'?plan.minimum?t(`剩余时间放不下下一道练习；最短组合约需 ${plan.minimum} 分钟。`,`The next exercise will not fit. The shortest available combination is about ${plan.minimum} min.`):t('当前可补充的练习暂缺估时，可逐题手动加入。','The available exercises have no confirmed time estimates. Add them individually.') :'';
 return <div className="interview-budget-body"><p>{t('先看建议，再加入练习组。按当前列表顺序挑选能放入预算的练习；已选题目和顺序保留。','Preview before adding. Pick exercises that fit in the current list order, keeping your existing choices and order.')}</p>
  <div className="interview-budget-controls"><label htmlFor={id+'-minutes'}>{t('可用时间（分钟）','Available time (minutes)')}<input id={id+'-minutes'} type="text" inputMode="numeric" maxLength={12} value={raw} aria-invalid={invalid||undefined} aria-describedby={id+'-feedback'} onChange={event=>setRaw(event.target.value)}/></label><fieldset><legend>{t('新增练习范围','Add exercises from')}</legend><label><input type="radio" name={id+'-mode'} value="remaining" checked={mode==='remaining'} disabled={!group.ready} onChange={()=>setMode('remaining')}/>{t('未通过的练习','Exercises not yet passed')}</label><label><input type="radio" name={id+'-mode'} value="all" checked={mode==='all'} onChange={()=>setMode('all')}/>{t('全部练习（含复习）','All exercises, including review')}</label></fieldset></div>
  <div className="interview-budget-presets" role="group" aria-label={t('快捷选择时间','Quick time budgets')}>{[30,60,90,120].map(value=><button key={value} type="button" aria-pressed={budget===value} onClick={()=>setRaw(String(value))}>{t(`${value} 分钟`,`${value} min`)}</button>)}</div>
  <div id={id+'-feedback'} className="interview-budget-feedback" role="status">{plan.reason?<p>{reason(plan)}</p>:<p>{t(`可补充 ${plan.additions.length} 道 · 整组约 ${plan.minutes} / ${budget} 分钟`,`Add ${plan.additions.length} exercises · About ${plan.minutes} of ${budget} min for the batch`)}</p>}{plan.selected.length>0&&plan.reason!=='estimate'&&plan.reason!=='unavailable'&&<small>{t(`已选 ${plan.selected.length} 道约 ${plan.selectedMinutes} 分钟已计入预算。`,`${plan.selected.length} selected exercises, about ${plan.selectedMinutes} min, are included in the budget.`)}</small>}{plan.unknownCandidates>0&&<small>{t(`${plan.unknownCandidates} 道可选练习缺少估时，未自动加入。`,`${plan.unknownCandidates} available exercises have no estimate and were not added automatically.`)}</small>}</div>
  {plan.additions.length>0&&<ol aria-label={t('待加入的时间安排','Proposed additions')}>{plan.additions.map(item=><li key={item.id}><span>{item.number}. {title(item.number)}</span><small>{t(`约 ${item.minutes} 分钟`,`About ${item.minutes} min`)}</small></li>)}</ol>}
  <button type="button" className="button subtle interview-budget-apply" aria-disabled={!!plan.reason||!plan.additions.length} onClick={()=>{if(!plan.reason&&plan.additions.length)group.applyPlan(plan);}}><Plus size={15} aria-hidden="true"/>{plan.additions.length&&!plan.reason?t(`加入这 ${plan.additions.length} 道练习`,`Add these ${plan.additions.length} exercises`):t('暂无可加入的安排','No additions available')}</button>
  <small className="interview-budget-disclaimer">{t('估时来自站内题目，实际用时因人而异；安排不改变成绩，也不代表公司考频。','Estimates come from the exercises; actual time varies. Planning does not change grades or represent company interview frequency.')}</small>
 </div>;
}
