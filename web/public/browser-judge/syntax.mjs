// CPython compilation only: learner statements are never executed here.
export const syntaxCheck = String.raw`
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
`;
