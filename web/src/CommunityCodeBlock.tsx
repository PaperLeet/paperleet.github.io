import {useEffect,useRef,useState} from 'react';
import {Check,Copy,WrapText} from 'lucide-react';
import {Highlight} from './SyntaxHighlight';
import {useI18n} from './i18n';
import './community-code-block.css';

type Props={code:string;language?:string};
type CopyStatus='idle'|'copying'|'copied'|'failed';

export default function CommunityCodeBlock({code,language='text'}:Props) {
  // A replaced snippet starts a new lifetime, including A → B → A replacements.
  return <CodeBlock key={JSON.stringify([code,language])} code={code} language={language}/>;
}

function CodeBlock({code,language}:Required<Props>) {
  const {t}=useI18n();
  const [wrapped,setWrapped]=useState(false),[status,setStatus]=useState<CopyStatus>('idle');
  const live=useRef(false),busy=useRef(false),request=useRef(0),deadline=useRef<number|undefined>(undefined),surface=useRef('');
  function retire() {
    request.current++;busy.current=false;window.clearTimeout(deadline.current);deadline.current=undefined;
  }
  useEffect(()=>{
    live.current=true;surface.current=location.href;
    const leave=()=>{
      if(surface.current===location.href)return;
      surface.current=location.href;retire();setStatus('idle');
    };
    window.addEventListener('hashchange',leave);window.addEventListener('popstate',leave);
    return()=>{live.current=false;retire();window.removeEventListener('hashchange',leave);window.removeEventListener('popstate',leave);};
  },[]);
  function copy() {
    if(busy.current)return;
    busy.current=true;setStatus('copying');
    const current=++request.current,href=location.href;surface.current=href;
    const settle=(next:CopyStatus)=>{
      if(!live.current||request.current!==current)return;
      retire();surface.current=location.href;setStatus(location.href===href?next:'idle');
    };
    deadline.current=window.setTimeout(()=>settle('failed'),8000);
    try {Promise.resolve(navigator.clipboard.writeText(code)).then(()=>settle('copied'),()=>settle('failed'));}
    catch {settle('failed');}
  }
  return <div className={`community-code-block${wrapped?' is-wrapped':''}`}>
    <div className="community-code-header">
      <span>{language}</span>
      <div className="community-code-actions">
        <button type="button" aria-pressed={wrapped} aria-label={t(`自动换行 ${language} 代码`,`Wrap ${language} code`)} onClick={()=>setWrapped(value=>!value)}>
          <WrapText size={15} aria-hidden="true"/>{t('自动换行','Wrap lines')}
        </button>
        <button type="button" aria-disabled={status==='copying'} aria-busy={status==='copying'} onClick={copy} aria-label={t(`复制 ${language} 代码`,`Copy ${language} code`)}>
          {status==='copied'?<Check size={15} aria-hidden="true"/>:<Copy size={15} aria-hidden="true"/>}
          {status==='copied'?t('已复制','Copied'):status==='copying'?t('正在复制…','Copying…'):t('复制代码','Copy code')}
        </button>
      </div>
    </div>
    <Highlight code={code} lang={language}/>
    <span className="sr-only" role="status">{status==='copying'?t('正在复制代码…','Copying code…'):status==='copied'?t('代码已复制。','Code copied.'):status==='failed'?t('自动复制不可用，请使用下方的手动复制区域。','Automatic copy is unavailable. Use the manual copy field below.'):null}</span>
    {status==='failed'&&<label className="community-code-manual">
      <span>{t('请选择下方代码，使用系统复制快捷键或菜单复制。','Select the code below and use your system’s copy shortcut or menu.')}</span>
      <textarea aria-label={t(`手动复制 ${language} 代码`,`Copy ${language} code manually`)} readOnly spellCheck={false} wrap="off" rows={Math.min(8,Math.max(2,code.split(/\r\n|\r|\n/).length))} value={code} onFocus={event=>event.currentTarget.select()}/>
    </label>}
  </div>;
}
