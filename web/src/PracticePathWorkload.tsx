import {Clock3} from 'lucide-react';
import {useI18n} from './i18n';
import {status,type Problem,type Saved} from './types';

function estimate(problems:Problem[]){
 let minutes=0,count=0;
 for(const problem of problems){const value=problem.estimated_minutes;if(typeof value==='number'&&Number.isFinite(value)&&value>0){minutes+=value;count++;}}
 return {minutes,count};
}

/** Published time estimates are planning aids; progress only determines unfinished work. */
export default function PracticePathWorkload({problems,total,saved,ready}:{problems:Problem[];total:number;saved:Record<string,Saved>;ready:boolean}){
 const {t,locale}=useI18n(),full=estimate(problems);
 if(!Number.isSafeInteger(total)||total<problems.length||!full.count||!Number.isFinite(full.minutes))return null;
 const remaining=ready?problems.filter(problem=>status(problem,saved[problem.id])!=='done'):[],unfinished=total-(problems.length-remaining.length),rest=estimate(remaining);
 const format=(minutes:number)=>new Intl.NumberFormat(locale==='en'?'en':'zh-CN',{maximumSignificantDigits:12}).format(minutes);
 const metrics=[{key:'full',value:full,total,label:full.count===total?t('全路径预计','Full path estimate'):t('全路径已知用时','Known path estimates')},...(ready&&unfinished>0?[{key:'remaining',value:rest,total:unfinished,label:rest.count===unfinished?t('未完成部分预计','Unfinished estimate'):t('未完成已知用时','Known unfinished estimates')}]:[])];
 return <dl className="practice-path-workload" aria-label={t('路径练习用时参考','Path practice time estimates')}>
  {metrics.map(metric=><div key={metric.key} data-path-estimate={metric.key}>
   <dt><Clock3 size={14} aria-hidden="true"/>{metric.label}</dt>
   <dd>{metric.value.count?t(`约 ${format(metric.value.minutes)} 分钟`,`About ${format(metric.value.minutes)} min`):t('待补全','Unavailable')}<small>{metric.value.count===metric.total?t(`${metric.total} 道练习`,`${metric.total} ${metric.total===1?'exercise':'exercises'}`):t(`已知 ${metric.value.count} / ${metric.total} 道练习`,`Known for ${metric.value.count} of ${metric.total} exercises`)}</small></dd>
  </div>)}
 </dl>;
}
