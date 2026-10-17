import {useId,useState} from 'react';
import {BookOpen,Code2,Users} from 'lucide-react';
import {useI18n} from './i18n';
import {ProblemSolutions} from './deferredPages';
import {Highlight} from './SyntaxHighlight';
import {navigateTabs} from './tabNavigation';
import type {ProblemTopic} from './communityTopics';
import './problem-solution-panel.css';

type Props={topic:ProblemTopic;solution?:string|null;communityAvailable:boolean;onCode:()=>void};

/** Keep a solution draft, feed filters and expanded replies alive while consulting the reference. */
export default function ProblemSolutionPanel(props:Props){
  return <SolutionPanel key={props.topic.id} {...props}/>;
}

function SolutionPanel({topic,solution,communityAvailable,onCode}:Props){
  const {t}=useI18n(),id=useId();
  const [view,setView]=useState<'community'|'reference'>(communityAvailable?'community':'reference');
  const active=communityAvailable?view:'reference';
  const [revealed,setRevealed]=useState(false);
  return <section className="solution-panel problem-solution-panel" aria-label={t('本题题解','Solutions for this problem')}>
    <div className="problem-solution-toolbar">
      <div className="problem-solution-tabs" role="tablist" aria-label={t('题解来源','Solution source')} onKeyDown={event=>{
        if(event.defaultPrevented||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
        navigateTabs(event);
      }}>
        {communityAvailable&&<button type="button" id={`${id}-community-tab`} role="tab" aria-selected={active==='community'} aria-controls={`${id}-community`} tabIndex={active==='community'?0:-1} onClick={()=>setView('community')}><Users size={16} aria-hidden="true"/>{t('社区题解','Community solutions')}</button>}
        <button type="button" id={`${id}-reference-tab`} role="tab" aria-selected={active==='reference'} aria-controls={`${id}-reference`} tabIndex={active==='reference'?0:-1} onClick={()=>setView('reference')}><BookOpen size={16} aria-hidden="true"/>{t('参考实现','Reference implementation')}</button>
      </div>
      <button type="button" className="problem-solution-code" onClick={onCode}><Code2 size={16} aria-hidden="true"/>{t('返回代码','Back to code')}</button>
    </div>
    {communityAvailable&&<div id={`${id}-community`} role="tabpanel" aria-labelledby={`${id}-community-tab`} tabIndex={0} hidden={active!=='community'}>
      <ProblemSolutions topic={topic} onCode={onCode}/>
    </div>}
    <div id={`${id}-reference`} className="reference-solution" role="tabpanel" aria-labelledby={`${id}-reference-tab`} tabIndex={0} hidden={active!=='reference'}>
      <h2>{t('参考实现','Reference implementation')}</h2>
      <p>{t('建议先独立尝试，用测试结果定位问题，再对照参考实现。','Try your own approach and use test results to locate issues before checking the reference.')}</p>
      <details open={revealed} onToggle={event=>setRevealed(event.currentTarget.open)}>
        <summary><BookOpen size={16} aria-hidden="true"/>{t('展开参考解（包含答案）','Reveal reference solution (spoilers)')}</summary>
        {revealed&&(solution?<Highlight code={solution}/>:<div className="notice">{t('参考实现尚未提供。','No reference implementation is available yet.')}</div>)}
      </details>
    </div>
  </section>;
}
