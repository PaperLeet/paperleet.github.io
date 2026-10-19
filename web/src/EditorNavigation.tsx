import {useEffect,useId,useRef,useState} from 'react';
import {ArrowDown,ArrowUp,Command,Search,X} from 'lucide-react';
import {EditorView} from '@codemirror/view';
import {gotoLine} from '@codemirror/search';
import {openEditorFind} from './editorSearch';
import {pythonDefinitions} from './pythonDefinitions';
import {useI18n} from './i18n';

type Props={mode:'symbols'|'shortcuts';view:EditorView;source:string;pending:boolean;readOnly:boolean;definitions:ReturnType<typeof pythonDefinitions>;onClose:(restore?:boolean)=>void;onNavigate:()=>void};
export default function EditorNavigation({mode,view,source,pending,readOnly,definitions,onClose,onNavigate}:Props){
  const {t}=useI18n(),[query,setQuery]=useState(''),[active,setActive]=useState(0),panel=useRef<HTMLDivElement>(null),input=useRef<HTMLInputElement>(null),list=useRef<HTMLDivElement>(null),id=useId();
  const rows=definitions.filter(row=>row.qualifiedName.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  const index=Math.min(active,Math.max(0,rows.length-1));
  useEffect(()=>{if(mode==='symbols')input.current?.focus();else panel.current?.querySelector<HTMLButtonElement>('button')?.focus();},[]);
  useEffect(()=>{const close=(event:PointerEvent)=>{if(panel.current&&!panel.current.contains(event.target as Node))onClose(false);};document.addEventListener('pointerdown',close);return()=>document.removeEventListener('pointerdown',close);},[onClose]);
  useEffect(()=>{list.current?.querySelector('[aria-selected=true]')?.scrollIntoView({block:'nearest'});},[index,query]);
  function jump(position:number){
    if(readOnly||pending||view.state.readOnly||view.state.doc.toString()!==source)return;
    const row=definitions.find(row=>row.from===position);if(!row||view.state.doc.sliceString(row.from,row.to)!==row.name)return;
    onNavigate();onClose(false);view.dispatch({selection:{anchor:row.from},effects:EditorView.scrollIntoView(row.from,{y:'center'})});view.focus();
  }
  const commands=[
    {label:t('查找','Find'),keys:'Ctrl / ⌘ F',action:()=>openEditorFind(view)},
    {label:t('查找和替换','Find and replace'),keys:'Ctrl H / ⌘ ⌥ F',action:()=>openEditorFind(view,true)},
    {label:t('跳转到行','Go to line'),keys:'Ctrl G',action:()=>gotoLine(view)},
  ];
  return <div ref={panel} className="python-editor-palette" role="dialog" aria-label={mode==='symbols'?t('跳转到函数或类','Go to function or class'):t('编辑器快捷键','Editor keyboard shortcuts')} onKeyDown={event=>{
    if(event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
    if(event.key==='Escape'&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey){event.preventDefault();event.stopPropagation();onClose();}
    if(mode!=='symbols'||event.target!==input.current)return;
    if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();setActive(rows.length?(index+(event.key==='ArrowDown'?1:-1)+rows.length)%rows.length:0);}
    if(event.key==='Enter'&&!event.altKey&&!event.ctrlKey&&!event.metaKey){event.preventDefault();if(rows[index])jump(rows[index].from);}
  }}>
    <header><span>{mode==='symbols'?<><Search size={14}/>{t('跳转到函数或类','Go to function or class')}</>:<><Command size={14}/>{t('编辑器快捷键','Editor keyboard shortcuts')}</>}</span><button type="button" className="icon-button" aria-label={t('关闭导航','Close navigation')} onClick={()=>onClose()}><X size={15}/></button></header>
    {mode==='symbols'?<>
      <input ref={input} role="combobox" aria-label={t('搜索函数或类','Search functions or classes')} aria-expanded="true" aria-autocomplete="list" aria-controls={id} aria-activedescendant={rows[index]?`${id}-${index}`:undefined} value={query} placeholder={t('输入函数或类名…','Type a function or class name…')} onChange={event=>{setQuery(event.target.value);setActive(0);}}/>
      <div ref={list} id={id} role="listbox" aria-label={t('函数和类','Functions and classes')}>
        {rows.map((row,i)=><button key={row.from} id={`${id}-${i}`} type="button" role="option" aria-selected={index===i} disabled={readOnly||pending} tabIndex={-1} onMouseDown={event=>event.preventDefault()} onClick={()=>jump(row.from)}><span className="python-symbol-kind">{row.kind==='class'?'C':'ƒ'}</span><span>{row.qualifiedName}</span><small>{t(`第 ${row.line} 行`,`line ${row.line}`)}</small></button>)}
        {(!rows.length||pending)&&<p role="status">{pending?t('正在更新…','Updating…'):t('没有匹配的函数或类','No matching functions or classes')}</p>}
      </div><footer><span><ArrowUp size={11}/><ArrowDown size={11}/>{t('选择','navigate')}</span><span>↵ {t('跳转','go')}</span><span>Esc {t('关闭','close')}</span></footer>
    </>:<>
      <div className="editor-shortcut-actions">{commands.map(command=><button type="button" key={command.keys} onClick={()=>{onClose(false);command.action();}}><span>{command.label}</span><kbd>{command.keys}</kbd></button>)}</div>
      <dl className="editor-shortcut-list">{[
        [t('查找下一个 / 上一个','Next / previous match'),'Enter / Shift Enter'],
        [t('跳转到函数或类','Go to function or class'),'Ctrl / ⌘ Shift O'],
        [t('正则 / 大小写 / 全词','Regex / case / whole word'),'Alt R / C / W · ⌘ ⌥ R / C / W'],
        [t('选中下一个相同内容','Select next occurrence'),'Ctrl / ⌘ D'],
        [t('切换行注释','Toggle line comment'),'Ctrl / ⌘ /'],
        [t('向上 / 向下移动行','Move line up / down'),'Alt ↑ / ↓'],
        [t('缩进 / 减少缩进','Indent / outdent'),'Tab / Shift Tab'],
        [t('运行测试','Run tests'),'Ctrl / ⌘ Enter'],
        [t('关闭浮层，返回代码','Close panel, return to code'),'Esc'],
      ].map(([label,keys])=><div key={keys}><dt>{label}</dt><dd><kbd>{keys}</kbd></dd></div>)}</dl>
    </>}
  </div>;
}
