import {useEffect,useRef,useState} from 'react';
import {useI18n} from './i18n';
import {parseCustomCase,type CustomCaseResult} from './customCaseModel';
import type {Problem} from './types';
import './custom-cases.css';

export default function CustomCases({problem,source,ready,busy,onBusyChange}:{problem:Problem;source:string;ready:boolean;busy:boolean;onBusyChange:(busy:boolean)=>void}) {
  const {t,locale}=useI18n(),names=problem.functions.map(row=>row.name);
  const [name,setName]=useState(names[0]||''),[args,setArgs]=useState('[]'),[kwargs,setKwargs]=useState('{}'),[expected,setExpected]=useState('');
  const [stage,setStage]=useState(''),[error,setError]=useState(''),[result,setResult]=useState<{source:string;input:string;value:CustomCaseResult}>();
  const controller=useRef<AbortController|undefined>(undefined);
  const active=!!stage;
  useEffect(()=>()=>{controller.current?.abort();onBusyChange(false);},[onBusyChange]);
  async function run(){
    if(active||busy||!ready)return;
    setError('');
    let customCase;
    try{customCase=parseCustomCase(name,args,kwargs,expected,names);}catch(cause){setError(t('输入无效：','Invalid input: ')+(cause instanceof Error?cause.message:String(cause)));return;}
    const next=new AbortController();controller.current=next;
    const snapshot=source;
    setResult(undefined);setStage('loading');onBusyChange(true);
    try {
      const {runBrowserCustomCase,browserReason}=await import('./browserJudge');
      next.signal.throwIfAborted();
      const outcome=await runBrowserCustomCase(problem,snapshot,customCase,next.signal,value=>{if(!next.signal.aborted)setStage(value);});
      if(next.signal.aborted)return;
      if(outcome.type==='custom_result')setResult({source:snapshot,input:JSON.stringify([name,args,kwargs,expected]),value:outcome.result});
      else if(outcome.type==='syntax_error')setError(outcome.failure.diagnostic.message);
      else if(outcome.type==='unsupported')setError(browserReason(outcome.reason,locale==='en')+(outcome.detail?'\n'+outcome.detail:''));
    }catch(cause){if(!next.signal.aborted)setError(String(cause));}
    finally{if(controller.current===next&&!next.signal.aborted){setStage('');onBusyChange(false);}}
  }
  function stop(){controller.current?.abort();setStage('');onBusyChange(false);setError(t('自定义运行已停止。','Custom run stopped.'));}
  return <details className="custom-cases"><summary>{t('自定义用例 / Console','Custom case / Console')}</summary>
    <p>{t('用自己的输入调用当前代码。自定义运行不计入通过记录或 Rating。','Call your current implementation with your own input. Custom runs do not affect completion or Rating.')}</p>
    {!problem.browser_judge?<p role="status">{t('此题暂无浏览器运行支持。','Browser execution is unavailable for this module.')}</p>:<>
      <label>{t('函数','Function')}<select aria-label={t('函数','Function')} value={name} disabled={active} onChange={event=>setName(event.target.value)}>{names.map(value=><option key={value}>{value}</option>)}</select></label>
      <label>{t('位置参数（JSON 数组）','Positional arguments (JSON array)')}<textarea aria-label={t('位置参数（JSON 数组）','Positional arguments (JSON array)')} value={args} maxLength={32768} rows={3} spellCheck={false} disabled={active} onChange={event=>setArgs(event.target.value)}/></label>
      <label>{t('关键字参数（JSON 对象）','Keyword arguments (JSON object)')}<textarea aria-label={t('关键字参数（JSON 对象）','Keyword arguments (JSON object)')} value={kwargs} maxLength={32768} rows={2} spellCheck={false} disabled={active} onChange={event=>setKwargs(event.target.value)}/></label>
      <label>{t('预期结果（可选 JSON）','Expected result (optional JSON)')}<textarea aria-label={t('预期结果（可选 JSON）','Expected result (optional JSON)')} value={expected} maxLength={32768} rows={2} spellCheck={false} disabled={active} onChange={event=>setExpected(event.target.value)}/></label>
      <details><summary>{t('如何传入张量？','How do I pass a tensor?')}</summary><p>{t('使用 __tensor__ 对象，dtype 可选 float32、float64、int64 或 bool。结果会转成 JSON；数字比较使用 atol=1e-6、rtol=1e-5。','Use a __tensor__ object, with float32, float64, int64 or bool dtype. Results become JSON; numeric comparison uses atol=1e-6 and rtol=1e-5.')}</p><pre>{'[{"__tensor__": [[1, 2], [3, 4]], "dtype": "float64"}]'}</pre></details>
      <div className="custom-case-actions"><button type="button" className="button" disabled={!ready||busy||active||!names.length} onClick={()=>void run()}>{active?stage==='running'?t('执行中…','Executing…'):t('准备运行时…','Preparing runtime…'):t('运行自定义用例','Run custom case')}</button>{active&&<button type="button" className="button subtle" onClick={stop}>{t('停止','Stop')}</button>}</div>
      <small>{t('执行限时 10 秒；首次需下载运行时。输出最多 16,000 字符，结果最多 10,000 个张量元素。','Execution limit: 10 seconds; the runtime downloads on first use. Output: 16,000 characters; tensor result: 10,000 elements.')}</small>
    </>}
    {error&&<pre role="alert">{error}</pre>}
    {result&&<section aria-label={t('自定义运行结果','Custom run result')} aria-live="polite">
      {(result.source!==source||result.input!==JSON.stringify([name,args,kwargs,expected]))&&<p className="notice">{t('代码或输入已修改，以下为之前运行的输出。','Code or input changed. The output below belongs to the previous run.')}</p>}
      <strong>{!result.value.ok?t('执行失败','Execution failed'):result.value.matches===false?t('与预期不同','Does not match expected'):result.value.matches===true?t('与预期一致','Matches expected'):t('运行完成','Run complete')}</strong>
      {'output' in result.value&&<pre>{JSON.stringify(result.value.output,null,2)}</pre>}
      {result.value.error&&<pre>{result.value.error}</pre>}
      <h4>Console</h4><pre>{result.value.stdout||t('没有控制台输出。','No console output.')}</pre>
    </section>}
  </details>;
}
