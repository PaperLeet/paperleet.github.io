// This code executes exclusively in the disposable opaque-origin worker.
export const customCasePython = String.raw`
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
`;
