import { useEffect, useState } from 'react';
import CodeMirror, { EditorView } from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import type { PythonEditorProps } from './EditorFallback';
import { editorSource, sourceOffset } from './sourceLocation';

const extensions = [python()];
export default function PythonEditor({ value, readOnly, theme, onChange, font, label, focusRequest, onFocusHandled }: PythonEditorProps) {
  const [view, setView] = useState<EditorView>();
  useEffect(() => {
    if (!view || !focusRequest || readOnly || focusRequest.source !== value) return;
    const source = editorSource(value);
    if (view.state.doc.toString() !== source) return;
    const position = sourceOffset(source, focusRequest.location);
    if (position !== undefined) {
      view.dom.scrollIntoView({ block: 'nearest' });
      view.dispatch({ selection: { anchor: position }, effects: EditorView.scrollIntoView(position, { y: 'center' }) });
      view.focus();
    }
    onFocusHandled?.(focusRequest);
  }, [view, value, readOnly, focusRequest, onFocusHandled]);
  return <CodeMirror readOnly={readOnly} value={value} height="100%" theme={theme} extensions={extensions}
    onCreateEditor={setView}
    onChange={onChange} basicSetup={{ lineNumbers: true, foldGutter: true, highlightActiveLine: true, autocompletion: true }}
    style={{ fontSize: font }} aria-label={label} />;
}
