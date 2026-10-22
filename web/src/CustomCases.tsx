import {useEffect,useId,useRef,useState} from 'react';
import {useI18n} from './i18n';
import {CustomCaseInputError,parseCustomCase,type CustomCaseField,type CustomCaseResult} from './customCaseModel';
import type {Problem} from './types';
import './custom-cases.css';

export default function CustomCases({problem,source,ready,busy,onBusyChange}:{problem:Problem;source:string;ready:boolean;busy:boolean;onBusyChange:(busy:boolean)=>void}) {
  const {t,locale}=useI18n(),names=problem.functions.map(row=>row.name);
  const [name,setName]=useState(names[0]||''),[args,setArgs]=useState('[]'),[kwargs,setKwargs]=useState('{}'),[expected,setExpected]=useState('');
  const [stage,setStage]=useState(''),[error,setError]=useState(''),[result,setResult]=useState<{source:string;input:string;value:CustomCaseResult}>();
  const [invalid,setInvalid]=useState<CustomCaseInputError>();
  const id=useId(),root=useRef<HTMLDetailsElement>(null);
  const controller=useRef<AbortController|undefined>(undefined);
  const active=!!stage;
  const invalidText=invalid&&(invalid.reason==='function'?t('请选择当前题目的函数。','Choose a function from this module.'):invalid.reason==='size'?t('所有输入合计不能超过 32 KB。','Combined input must be at most 32 KB.'):invalid.reason==='array'?t('位置参数应为 JSON 数组，例如 []。','Positional arguments must be a JSON array, such as [].'):invalid.reason==='object'?t('关键字参数应为 JSON 对象，例如 {}。','Keyword arguments must be a JSON object, such as {}.'):t('请输入有效的 JSON，字符串请使用双引号。','Enter valid JSON. Use double quotes for strings.'));
  function fieldProps(field:CustomCaseField){return {'data-field':field,'aria-invalid':invalid?.field===field||undefined,'aria-describedby':invalid?.field===field?id+'-error':undefined};}
  function fieldError(field:CustomCaseField){return invalid?.field===field?<span className="custom-case-error" id={id+'-error'} role="alert">{invalidText}</span>:null;}
  useEffect(()=>{if(invalid){const target=root.current?.querySelector<HTMLElement>(`[data-field="${invalid.field}"]`);target?.focus({preventScroll:true});target?.closest('label')?.scrollIntoView?.({block:'nearest'});}},[invalid]);
  function edited(field:CustomCaseField){if(invalid?.field===field||invalid?.reason==='size')setInvalid(undefined);}
  useEffect(()=>()=>{controller.current?.abort();onBusyChange(false);},[onBusyChange]);
  async function run(){
    if(active||busy||!ready)return;
    setError('');setInvalid(undefined);
    let customCase;
    try{customCase=parseCustomCase(name,args,kwargs,expected,names);}catch(cause){if(cause instanceof CustomCaseInputError){setInvalid(cause);}else setError(t('无法读取输入，请检查 JSON 格式。','Unable to read input. Check the JSON format.'));return;}
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
  return <details className="custom-cases" ref={root}><summary>{t('自定义用例 / Console','Custom case / Console')}</summary>
    <p>{t('用自己的输入调用当前代码。自定义运行不计入通过记录或 Rating。','Call your current implementation with your own input. Custom runs do not affect completion or Rating.')}</p>
    {!problem.browser_judge?<p role="status">{t('此题暂无浏览器运行支持。','Browser execution is unavailable for this module.')}</p>:<>
      <label>{t('函数','Function')}<select {...fieldProps('function')} aria-label={t('函数','Function')} value={name} disabled={active} onChange={event=>{setName(event.target.value);edited('function');}}>{names.map(value=><option key={value}>{value}</option>)}</select>{fieldError('function')}</label>
      <label>{t('位置参数（JSON 数组）','Positional arguments (JSON array)')}<textarea {...fieldProps('args')} aria-label={t('位置参数（JSON 数组）','Positional arguments (JSON array)')} value={args} maxLength={32768} rows={3} spellCheck={false} disabled={active} onChange={event=>{setArgs(event.target.value);edited('args');}}/>{fieldError('args')}</label>
      <label>{t('关键字参数（JSON 对象）','Keyword arguments (JSON object)')}<textarea {...fieldProps('kwargs')} aria-label={t('关键字参数（JSON 对象）','Keyword arguments (JSON object)')} value={kwargs} maxLength={32768} rows={2} spellCheck={false} disabled={active} onChange={event=>{setKwargs(event.target.value);edited('kwargs');}}/>{fieldError('kwargs')}</label>
      <label>{t('预期结果（可选 JSON）','Expected result (optional JSON)')}<textarea {...fieldProps('expected')} aria-label={t('预期结果（可选 JSON）','Expected result (optional JSON)')} value={expected} maxLength={32768} rows={2} spellCheck={false} disabled={active} onChange={event=>{setExpected(event.target.value);edited('expected');}}/>{fieldError('expected')}</label>
      <details><summary>{t('如何传入张量？','How do I pass a tensor?')}</summary><p>{t('使用 __tensor__ 对象，dtype 可选 float32、float64、int64 或 bool。结果会转成 JSON；数字比较使用 atol=1e-6、rtol=1e-5。','Use a __tensor__ object, with float32, float64, int64 or bool dtype. Results become JSON; numeric comparison uses atol=1e-6 and rtol=1e-5.')}</p><pre>{'[{"__tensor__": [[1, 2], [3, 4]], "dtype": "float64"}]'}</pre></details>
      <div className="custom-case-actions"><button type="button" className="button" aria-busy={active} aria-describedby={id+'-status'} disabled={!ready||busy||active||!names.length} onClick={()=>void run()}>{active?stage==='running'?t('执行中…','Executing…'):t('准备运行时…','Preparing runtime…'):t('运行自定义用例','Run custom case')}</button>{active&&<button type="button" className="button subtle" onClick={stop}>{t('停止','Stop')}</button>}</div>
      <p id={id+'-status'} role="status">{active?stage==='running'?t('正在执行当前代码，可随时停止。','Running your current code. You can stop at any time.'):t('正在准备 Python 运行时，首次运行需要下载。','Preparing the Python runtime. A download is needed on first use.'):!ready?t('正在准备工作区，完成后即可运行。','Preparing your workspace. Run when it is ready.'):busy?t('请等待当前测试完成后再运行自定义用例。','Wait for the current test to finish before running a custom case.'):!names.length?t('当前题目没有可调用的函数。','This module has no callable functions.'):invalid?t('请修正标记的输入后重试。','Correct the highlighted input and try again.'):t('准备就绪。','Ready to run.')}</p>
      <small>{t('执行限时 10 秒；首次需下载运行时。输出最多 16,000 字符，结果最多 10,000 个张量元素。','Execution limit: 10 seconds; the runtime downloads on first use. Output: 16,000 characters; tensor result: 10,000 elements.')}</small>
    </>}
    {error&&<pre role="alert">{error}</pre>}
    {result&&<section className="custom-case-result" data-state={!result.value.ok?'error':result.value.matches===false?'mismatch':result.value.matches===true?'match':'done'} aria-label={t('自定义运行结果','Custom run result')} aria-live="polite">
      {(result.source!==source||result.input!==JSON.stringify([name,args,kwargs,expected]))&&<p className="notice">{t('代码或输入已修改，以下为之前运行的输出。','Code or input changed. The output below belongs to the previous run.')}</p>}
      <strong>{!result.value.ok?t('执行失败','Execution failed'):result.value.matches===false?t('与预期不同','Does not match expected'):result.value.matches===true?t('与预期一致','Matches expected'):t('运行完成','Run complete')}</strong>
      {'output' in result.value&&<><h4>{t('输出','Output')}</h4><pre>{JSON.stringify(result.value.output,null,2)}</pre></>}
      {result.value.error&&<pre>{result.value.error}</pre>}
      <h4>Console</h4><pre>{result.value.stdout||t('没有控制台输出。','No console output.')}</pre>
    </section>}
  </details>;
}
