import { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from 'react';
import CodeMirror, { EditorView } from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import type { PythonEditorProps } from './EditorFallback';
import { editorSource, sourceOffset } from './sourceLocation';
import { pythonDefinitions } from './pythonDefinitions';
import { useI18n } from './i18n';
import './python-editor.css';

const pythonLanguage = python();
const editorSetup = { lineNumbers: true, foldGutter: true, highlightActiveLine: true, autocompletion: true };
export default function PythonEditor({ value, readOnly, theme, onChange, font, label, focusRequest, onFocusHandled }: PythonEditorProps) {
  const { t } = useI18n();
  const pickerId = useId();
  const [view, setView] = useState<EditorView>();
  const [choice, setChoice] = useState<{ source: string; from: number }>();
  const handledFocus = useRef<typeof focusRequest>(undefined);
  const source = useMemo(() => editorSource(value), [value]);
  const outlineSource = useDeferredValue(source);
  const definitions = useMemo(() => pythonDefinitions(outlineSource), [outlineSource]);
  const selected = choice?.source === source && outlineSource === source ? definitions.find(item => item.from === choice.from) : undefined;
  const extensions = useMemo(() => [pythonLanguage, EditorView.contentAttributes.of({ 'aria-label': label })], [label]);
  useEffect(() => {
    if (!view || !focusRequest || handledFocus.current === focusRequest || readOnly || focusRequest.source !== value) return;
    if (view.state.doc.toString() !== source) return;
    handledFocus.current = focusRequest;
    const allowed=!focusRequest.claimFocus||focusRequest.claimFocus();
    const position=focusRequest.location?sourceOffset(source,focusRequest.location):undefined;
    let focused=false;
    if(allowed&&(!focusRequest.location||position!==undefined)) {
      if(focusRequest.location&&position!==undefined) {
        view.dom.scrollIntoView({block:'nearest'});
        view.dispatch({selection:{anchor:position},effects:EditorView.scrollIntoView(position,{y:'center'})});
      }
      view.focus();
      focused=view.contentDOM.ownerDocument.activeElement===view.contentDOM;
    }
    onFocusHandled?.(focusRequest,focused);
  }, [view, value, source, readOnly, focusRequest, onFocusHandled]);
  function acknowledgeFocusRequest() {
    if (focusRequest && handledFocus.current !== focusRequest) { handledFocus.current = focusRequest; onFocusHandled?.(focusRequest); }
  }
  function chooseDefinition(position: string) {
    if (!view || readOnly || outlineSource !== source || view.state.doc.toString() !== outlineSource) return;
    const definition = definitions.find(item => String(item.from) === position);
    acknowledgeFocusRequest();
    setChoice(definition ? { source, from: definition.from } : undefined);
  }
  function goToDefinition() {
    if (!view || readOnly || outlineSource !== source || view.state.doc.toString() !== outlineSource) return;
    const definition = selected;
    if (!definition || view.state.doc.sliceString(definition.from, definition.to) !== definition.name) return;
    // A newer explicit navigation takes ownership from an older diagnostic request.
    acknowledgeFocusRequest();
    view.dom.scrollIntoView({ block: 'nearest' });
    view.dispatch({ selection: { anchor: definition.from }, effects: EditorView.scrollIntoView(definition.from, { y: 'center' }) });
    view.focus();
  }
  return <div className="python-editor">
    <div className="python-definition-nav">
      <label htmlFor={pickerId}>{t('代码导航', 'Code navigation')}</label>
      <select id={pickerId} aria-label={t('选择要定位的函数或类', 'Choose a function or class')} value={selected ? String(selected.from) : ''}
        disabled={readOnly || !view || outlineSource !== source || definitions.length === 0}
        onChange={event => chooseDefinition(event.target.value)}>
        <option value="">{outlineSource !== source ? t('正在更新…', 'Updating…') : definitions.length ? t(`函数和类（${definitions.length}）`, `Functions and classes (${definitions.length})`) : t('暂无函数或类', 'No functions or classes')}</option>
        {definitions.map(definition => <option key={definition.from} value={String(definition.from)}>
          {definition.kind === 'class' ? 'class ' : ''}{definition.qualifiedName} · {t(`第 ${definition.line} 行`, `line ${definition.line}`)}
        </option>)}
      </select>
      <button type="button" disabled={readOnly || !view || !selected} onClick={goToDefinition}
        aria-label={t('定位到所选定义', 'Go to selected definition')}>{t('定位', 'Go')}</button>
    </div>
    <CodeMirror readOnly={readOnly} value={value} height="100%" theme={theme} extensions={extensions}
      onCreateEditor={setView}
      onChange={onChange} basicSetup={editorSetup}
      style={{ fontSize: font }} aria-label={label} />
  </div>;
}
