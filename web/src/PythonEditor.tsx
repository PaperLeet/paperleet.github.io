import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import CodeMirror, { EditorView } from '@uiw/react-codemirror';
import {Prec} from '@codemirror/state';
import {keymap} from '@codemirror/view';
import {gotoLine,closeSearchPanel} from '@codemirror/search';
import {editorSearch,editorSearchLocale,openEditorFind} from './editorSearch';
import EditorNavigation from './EditorNavigation';
import { python } from '@codemirror/lang-python';
import type { PythonEditorProps } from './EditorFallback';
import { editorSource, sourceOffset } from './sourceLocation';
import { pythonDefinitions } from './pythonDefinitions';
import { useI18n } from './i18n';
import './python-editor.css';

const pythonLanguage = python();
const editorSetup = { lineNumbers: true, foldGutter: true, highlightActiveLine: true, autocompletion: true, searchKeymap:false };
export default function PythonEditor({ value, readOnly, theme, onChange, font, label, focusRequest, onFocusHandled }: PythonEditorProps) {
  const {locale} = useI18n();
  const root=useRef<HTMLDivElement>(null);
  const [palette,setPalette]=useState<'symbols'|'shortcuts'>();
  const [view, setView] = useState<EditorView>();
  const handledFocus = useRef<typeof focusRequest>(undefined);
  const source = useMemo(() => editorSource(value), [value]);
  const outlineSource = useDeferredValue(source);
  const definitions = useMemo(() => pythonDefinitions(outlineSource), [outlineSource]);
  const actions=useRef({symbols:()=>setPalette('symbols'),shortcuts:()=>setPalette('shortcuts')});
  const extensions = useMemo(() => [pythonLanguage,editorSearch,editorSearchLocale(locale),
    Prec.high(keymap.of([
      {key:'Mod-Shift-o',scope:'editor search-panel',run:view=>{closeSearchPanel(view);actions.current.symbols();return true;}},
      {key:'F1',scope:'editor search-panel',run:view=>{closeSearchPanel(view);actions.current.shortcuts();return true;}},
      {key:'Mod-Shift-p',scope:'editor search-panel',run:view=>{closeSearchPanel(view);actions.current.shortcuts();return true;}},
    ])),EditorView.contentAttributes.of({ 'aria-label': label })], [label,locale]);
  useEffect(()=>{
    const node=root.current;if(!node||!view)return;
    const command=(event:Event)=>{
      const name=(event as CustomEvent).detail;
      if(!['find','replace','line','symbols','shortcuts'].includes(name))return;
      setPalette(undefined);
      if(name==='find'||name==='replace')openEditorFind(view,name==='replace');
      if(name==='line')gotoLine(view);
      if(name==='symbols'||name==='shortcuts'){closeSearchPanel(view);setPalette(name);}
    };
    node.addEventListener('paperleet:editor-command',command);return()=>node.removeEventListener('paperleet:editor-command',command);
  },[view]);
  useEffect(() => {
    if (!view || !focusRequest || handledFocus.current === focusRequest || readOnly || focusRequest.source !== value) return;
    if (view.state.doc.toString() !== source) return;
    handledFocus.current = focusRequest;
    const allowed=!focusRequest.claimFocus||focusRequest.claimFocus();
    const position=focusRequest.location?sourceOffset(source,focusRequest.location):undefined;
    let focused=false;
    if(allowed&&(!focusRequest.location||position!==undefined)) {
      setPalette(undefined);
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
  return <div className="python-editor" ref={root}>
    {palette&&view&&<EditorNavigation key={palette} mode={palette} view={view} source={outlineSource} pending={outlineSource!==source} readOnly={readOnly} definitions={definitions} onNavigate={acknowledgeFocusRequest} onClose={(restore=true)=>{setPalette(undefined);if(restore)view.focus();}}/>}
    <CodeMirror readOnly={readOnly} value={value} height="100%" theme={theme} extensions={extensions}
      onCreateEditor={setView}
      onChange={onChange} basicSetup={editorSetup}
      style={{ fontSize: font }} aria-label={label} />
  </div>;
}
