import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import type { PythonEditorProps } from './EditorFallback';

const extensions = [python()];
export default function PythonEditor({ value, readOnly, theme, onChange, font, label }: PythonEditorProps) {
  return <CodeMirror readOnly={readOnly} value={value} height="100%" theme={theme} extensions={extensions}
    onChange={onChange} basicSetup={{ lineNumbers: true, foldGutter: true, highlightActiveLine: true, autocompletion: true }}
    style={{ fontSize: font }} aria-label={label} />;
}
