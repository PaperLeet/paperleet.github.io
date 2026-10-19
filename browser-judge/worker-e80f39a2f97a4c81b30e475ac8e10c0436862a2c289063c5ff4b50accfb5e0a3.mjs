function E(e,t){let r={python:"Python runtime",syntax:"Syntax check",packages:"NumPy and test packages",preparing:"Test environment",calibration:"Reference and mutation calibration",tests:"Learner tests"},n=(Object.hasOwn(r,t)?`Stage: ${r[t]}
`:"")+String(e);return n.length<=4e3?n:`${n.slice(0,500)}
\u2026
${n.slice(-3490)}`}function A(e){return e.tests.some(t=>Object.hasOwn(t,"unsupported_api")?t.unsupported_api===!0:/BrowserUnsupported|ModuleNotFoundError|ImportError/.test(t.message))}function $(e){let t=[String(e?.reason||"unknown_calibration_failure")];e?.mutant&&t.push(`mutant: ${e.mutant}`);let r=e?.reference||e?.report;for(let a of Array.isArray(r?.tests)?r.tests:[])a.passed||t.push(`${a.id}: ${String(a.message||a.outcome||"failed").slice(-1e3)}`);return r?.error&&t.push(String(r.error).slice(-1e3)),t.join(`
`).slice(0,4e3)}var C=String.raw`
import json
def _paperleet_syntax(source):
    try:
        compile(source, 'submission.py', 'exec')
    except SyntaxError as error:
        return {'kind': type(error).__name__, 'message': str(error.msg)[:500],
                'line': error.lineno or 1, 'column': error.offset or 1}
    return None
_paperleet_syntax_result = _paperleet_syntax(learner_source)
json.dumps(_paperleet_syntax_result, ensure_ascii=False)
`;var q=["numpy","atomicwrites","attrs","six","more-itertools","pluggy","py","setuptools","pyparsing","iniconfig","exceptiongroup","pytest"];async function R(e,t=[]){let r=typeof t=="boolean"?t?["pyyaml"]:[]:t;if(!Array.isArray(r)||r.some(c=>c!=="pyyaml")||new Set(r).size!==r.length)throw Error("Unvalidated browser packages");let a=[...q,...r],n="";for(let c=0;c<2;c++)try{await e.loadPackage(a,{errorCallback(i){n=String(i).slice(-2e3)}});let o=a.filter(i=>!Object.hasOwn(e.loadedPackages,i));if(o.length)throw Error(`Missing browser packages: ${o.join(", ")}`);e.runPython(`import numpy, pytest${r.includes("pyyaml")?", yaml":""}`);return}catch(o){n=`${String(o)}${n?`
${n}`:""}`.slice(-3e3)}throw Error(`Browser judge dependencies could not be loaded. ${n}`)}var _=e=>!!e&&typeof e=="object"&&!Array.isArray(e);function v(e,t){if(!_(e)||!_(t.tests)||!_(t.mutants))return;let r=Object.values(t.tests);if(!r.length||!r.every(o=>Number.isSafeInteger(o)&&Number(o)>=0))return;let a=r.reduce((o,i)=>o+Number(i),0),n=Object.keys(t.mutants);if(a<1||e.reference_passed!==a||!_(e.mutants_caught)||Object.keys(e.mutants_caught).length!==n.length)return;let c={};for(let o of n){if(!Object.hasOwn(e.mutants_caught,o))return;let i=e.mutants_caught[o];if(!Array.isArray(i)||!i.length||i.length>a||!i.every(u=>typeof u=="string"&&u.length>0&&u.length<=500))return;Object.defineProperty(c,o,{value:[...i],enumerable:!0,configurable:!0,writable:!0})}return{reference_passed:a,mutants_caught:c}}function M(e,t,r,a){if(!t||!_(e)||e.key!==t||typeof e.environment!="string"||!e.environment.length||e.environment.length>1e3||a!==void 0&&e.environment!==a)return;let n=v(e.summary,r);return n?{key:t,environment:e.environment,summary:n}:void 0}var F=String.raw`
import contextlib, io, json, math, time, traceback

def _paperleet_custom_run(source, raw):
    class BoundedOutput(io.TextIOBase):
        def __init__(self): self.text = ''; self.truncated = False
        def write(self, value):
            remaining = max(0, 16000 - len(self.text))
            self.text += value[:remaining]
            if len(value) > remaining: self.truncated = True
            return len(value)
        def flush(self): pass
    sink = BoundedOutput()
    def convert(value, depth=0):
        if depth > 30: raise ValueError('Input is too deeply nested')
        if isinstance(value, list): return [convert(item, depth+1) for item in value]
        if isinstance(value, dict):
            if '__tensor__' in value:
                import torch
                if set(value) - {'__tensor__','dtype','requires_grad'}: raise ValueError('Unknown tensor option')
                dtype = value.get('dtype', 'float64')
                if dtype not in ('float32','float64','int64','bool'): raise ValueError('Unsupported tensor dtype')
                return torch.tensor(value['__tensor__'], dtype=getattr(torch,dtype), requires_grad=value.get('requires_grad',False))
            return {key:convert(item,depth+1) for key,item in value.items()}
        return value
    visited = [0]
    def encode(value, depth=0):
        visited[0] += 1
        if depth > 30 or visited[0] > 12000: raise ValueError('Result exceeds the display limit')
        if value is None or isinstance(value,(str,bool,int)): return value
        if isinstance(value,float): return value if math.isfinite(value) else str(value)
        if isinstance(value,(tuple,list)): return [encode(item,depth+1) for item in value]
        if isinstance(value,dict): return {str(key):encode(item,depth+1) for key,item in value.items()}
        if hasattr(value,'numel') and value.numel()>10000: raise ValueError('Tensor result exceeds 10000 elements')
        if hasattr(value,'size') and isinstance(value.size,int) and value.size>10000: raise ValueError('Array result exceeds 10000 elements')
        if hasattr(value,'tolist'): return encode(value.tolist(),depth+1)
        if hasattr(value,'item'): return encode(value.item(),depth+1)
        raise TypeError('Return JSON-compatible values or tensors/arrays')
    def matches(actual, expected):
        if isinstance(actual,bool) or isinstance(expected,bool): return type(actual) is type(expected) and actual==expected
        if isinstance(actual,(int,float)) and isinstance(expected,(int,float)): return math.isclose(actual,expected,rel_tol=1e-5,abs_tol=1e-6)
        if isinstance(actual,list) and isinstance(expected,list): return len(actual)==len(expected) and all(matches(a,b) for a,b in zip(actual,expected))
        if isinstance(actual,dict) and isinstance(expected,dict): return actual.keys()==expected.keys() and all(matches(actual[key],expected[key]) for key in actual)
        return type(actual) is type(expected) and actual==expected
    result = {}
    try:
        case = json.loads(raw)
        namespace = {'__name__':'__paperleet_custom__'}
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(source,'learner.py','exec'),namespace)
            function = namespace.get(case['function'])
            if not callable(function): raise ValueError('The selected function is missing or not callable')
            output = encode(function(*convert(case['args']),**convert(case['kwargs'])))
        result = {'ok':True,'output':output}
        if 'expected' in case: result['matches'] = matches(output,case['expected'])
    except BaseException:
        result = {'ok':False,'error':traceback.format_exc()[-4000:]}
    result['stdout'] = sink.text + ('\n[Output truncated]' if sink.truncated else '')
    encoded = json.dumps(result,ensure_ascii=True,allow_nan=False)
    if len(encoded)>95000:
        encoded = json.dumps({'ok':False,'error':'Result exceeds the display limit','stdout':result['stdout']})
    return encoded
_paperleet_custom_run(learner_source,custom_case_json)
`;var m=globalThis.postMessage.bind(globalThis),J=!1;self.onmessage=async({data:e})=>{if(J)return;J=!0;let{base:t,bundle:r,source:a,calibrationKey:n="",cachedCalibration:c,customCase:o}=e,i=performance.now(),u=(d,s={})=>m({type:"progress",stage:d,...s}),f=(d,s="")=>m({type:"unsupported",reason:d,detail:s}),p="python";try{u("loading",{step:"python"});let{loadPyodide:d}=await import(`${t}browser-runtime/pyodide-0.29.4/pyodide.mjs`),s=await d({indexURL:`${t}browser-runtime/pyodide-0.29.4/`,stdout(){},stderr(){}});p="syntax",u("loading",{step:"syntax"}),s.globals.set("learner_source",a);let k=JSON.parse(s.runPython(C));if(k)return m({type:"syntax_error",diagnostic:k});if(p="packages",u("loading",{step:"packages"}),await R(s,r.version===3?r.packages:Object.keys(r.files||{}).some(l=>l.endsWith("/problem.yaml"))),p="preparing",u("loading",{step:"preparing"}),s.globals.set("bundle_json",JSON.stringify(r)),s.runPython(`import json, inspect
bundle = json.loads(bundle_json)
exec(compile(bundle['harness'], 'browser_harness.py', 'exec'))
prepare(bundle, bundle['adapter'], '/judge')`),o){p="custom",u("running"),s.globals.set("custom_case_json",JSON.stringify(o));let l=JSON.parse(s.runPython(F));m({type:"custom_result",result:l});return}s.globals.set("_browser_progress",(l,x,w)=>u(l==="tests"?"running":"calibrating",{step:l,completed:x,total:w})),s.globals.set("_browser_activity",(l,x,w,I,V,z,U,B)=>u(l==="tests"?"running":"calibrating",{step:l,completed:x,total:w,activity:{index:I,test:V,layer:z,why:U,...B?{mutant:B}:{}}}));let T=s.runPython("'progress' in inspect.signature(calibrate).parameters and 'progress' in inspect.signature(run_suite).parameters"),D=s.runPython("'activity' in inspect.signature(calibrate).parameters and 'activity' in inspect.signature(run_suite).parameters"),j=(T?", progress=_browser_progress":"")+(D?", activity=_browser_activity":""),S=JSON.stringify([s.version,s.runPython("json.dumps([__import__('sys').version, __import__('numpy').__version__, __import__('sys').byteorder])")]),g=M(c,n,r,S),N=performance.now();p="calibration",u("calibrating");let y=g?{compatible:!0,...g.summary}:JSON.parse(s.runPython(`json.dumps(calibrate(bundle, '/judge'${j}), allow_nan=False)`));if(!y.compatible)return f("calibration_failed",$(y));let b=v(y,r);if(!b)return f("calibration_failed","Incomplete reference/mutation calibration");m({type:"calibrated",proof:{key:n,environment:S,summary:b},reused:!!g});let O=performance.now();p="tests",u("running");let h=JSON.parse(s.runPython(`json.dumps(run_suite(bundle, learner_source, '/judge'${j}), allow_nan=False)`));if(h.error)return f("collection_failed");if(A(h))return f("python_api_unsupported");let P=performance.now();m({type:"result",report:h,calibration:b,measurements:{loading_ms:N-i,calibration_ms:O-N,learner_ms:P-O,total_ms:P-i,reused:!!g}})}catch(d){f("runtime_failed",E(d,p))}};
