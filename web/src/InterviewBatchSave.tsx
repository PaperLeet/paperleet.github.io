import AppLink from './AppLink';
import {useEffect,useRef,useState} from 'react';
import {ArrowRight,CheckCircle2,FolderPlus} from 'lucide-react';
import {useI18n} from './i18n';
import {useStudyLists} from './useStudyLists';
import {StudyListFeedback} from './StudyListControls';
import {newStudyListId,STUDY_LIST_ITEM_LIMIT,STUDY_LIST_LIMIT,type StudyListOperation} from './studyListStore';
import {favoriteHref} from './libraryNavigation';
import './study-lists.css';
import './interview-batch-save.css';

export type StudyListModule={id:string;number:number;title:string;translations?:{en?:{title:string}}};

type Attempt={request:string;list:string;destination:string;title:string};
export default function InterviewBatchSave({capture,modules,initialTitle,canAct}:{capture:{ids:readonly string[]};modules:StudyListModule[];initialTitle:string;canAct?:()=>boolean}){
 const {t,locale}=useI18n(),{user,loading,capabilities,state,store}=useStudyLists(),root=useRef<HTMLElement>(null),heading=useRef<HTMLHeadingElement>(null),openedAt=useRef(location.href);
 const [destination,setDestination]=useState('new'),[title,setTitle]=useState(initialTitle),[attempt,setAttempt]=useState<Attempt>(),[confirmed,setConfirmed]=useState<string>();
 const target=state.snapshot?.lists.find(list=>list.id===destination),missing=capture.ids.filter(id=>!target?.items.some(item=>item.problem_id===id));
 const matches=!!attempt&&(confirmed===attempt.request||state.confirmation?.request_id===attempt.request&&state.confirmation.list_id===attempt.list);
 useEffect(()=>{if(attempt&&state.confirmation?.request_id===attempt.request&&state.confirmation.list_id===attempt.list)setConfirmed(attempt.request);},[attempt,state.confirmation]);
 const saved=matches?state.snapshot?.lists.find(list=>list.id===attempt!.list):undefined,complete=!!saved&&capture.ids.every(id=>saved.items.some(item=>item.problem_id===id));
 const sameIntent=complete&&matches&&attempt?.destination===destination&&(destination!=='new'||attempt.title===title.trim());
 const capacity=destination==='new'?!!state.snapshot&&state.snapshot.lists.length>=STUDY_LIST_LIMIT:!!target&&target.items.length+missing.length>STUDY_LIST_ITEM_LIMIT;
 const blocked=loading||capabilities?.study_lists!==true||!user||!!state.pending||state.loading||!!state.error||!state.snapshot||capacity||sameIntent||destination!=='new'&&(!target||!missing.length)||destination==='new'&&(!title.trim()||Array.from(title.trim()).length>60);
 function here(){return !!root.current?.isConnected&&!root.current.closest('[hidden],[inert],[aria-hidden="true"]')&&(canAct?canAct():location.href===openedAt.current);}
 function save(){
  if(blocked||!store||!here())return;
  const op:StudyListOperation=destination==='new'?{action:'create',list_id:newStudyListId(),title:title.trim(),problem_ids:[...capture.ids]}:{action:'add',list_id:destination,problem_ids:[...missing]};
  const before=store.getSnapshot().pending,result=store.mutate(op),pending=store.getSnapshot().pending;
  if(pending&&pending!==before&&pending.command.list_id===op.list_id){
   const request=pending.command.request_id;
   setAttempt({request,list:op.list_id,destination,title:title.trim()});
   // Another widget may confirm a later write before React renders this receipt.
   void result.then(ok=>{if(ok)setConfirmed(request);});
  }
 }
 return <section ref={root} className="interview-batch-save" aria-label={t('保存练习组到私人题单','Save this batch to a private study list')}>
  <h4 ref={heading} tabIndex={-1}><FolderPlus size={16} aria-hidden="true"/>{t(`保存这 ${capture.ids.length} 道练习`,`Save these ${capture.ids.length} exercises`)}</h4>
  <p>{t('仅保存题目与顺序。题单仅自己可见，代码和学习记录不会复制。','Saves problems and their order to your private account list. Code and practice records are not copied.')}</p>
  <ol>{modules.map(module=><li key={module.id}>{module.number}. {locale==='en'?module.translations?.en?.title||module.title:module.title}</li>)}</ol>
  <StudyListFeedback state={state} store={store} canAct={()=>here()} focus={()=>heading.current?.focus({preventScroll:true})}/>
  {!state.snapshot?<p role="status">{state.loading?t('正在读取题单…','Loading study lists…'):t('题单尚未读取。','Study lists have not loaded.')}</p>:<form onSubmit={event=>{event.preventDefault();save();}}>
   <label>{t('保存位置','Save to')}<select value={destination} onChange={event=>setDestination(event.target.value)}><option value="new">{t('新建私人题单','Create a private list')}</option>{state.snapshot.lists.map(list=><option key={list.id} value={list.id}>{list.title} · {list.items.length}</option>)}{destination!=='new'&&!target&&<option value={destination} disabled>{t('原题单已不可用，请重新选择','The selected list is unavailable; choose again')}</option>}</select></label>
   <div hidden={destination!=='new'}><label>{t('题单名称','Study-list name')}<input value={title} maxLength={120} onChange={event=>setTitle(event.target.value)} required={destination==='new'}/></label></div>
   {destination!=='new'&&<p>{t('原题单已有题目的位置不变；缺少的题目按上方顺序追加。','Existing problems keep their positions. Missing problems are appended in the order above.')}</p>}
   {capacity&&<p role="status">{destination==='new'?t('已达到 30 个题单上限，可加入已有题单。','You have reached the 30-list limit. Add to an existing list.'):t('此题单容量不足，请选择其他题单或新建。','This list has insufficient room. Choose another list or create one.')}</p>}
   <button type="submit" className="button subtle" disabled={blocked}><FolderPlus size={15} aria-hidden="true"/>{sameIntent?t('保存已确认','Save confirmed'):destination==='new'?t('保存为新题单','Save as a new list'):!missing.length&&target?t('这组题已在题单中','All problems are already in this list'):t(`追加 ${missing.length} 道题`,`Append ${missing.length} problems`)}</button>
   {destination!=='new'&&target&&!missing.length&&<AppLink className="text-button" href={favoriteHref({query:'',state:'all',list:target.id})}>{t('打开题单','Open list')}<ArrowRight size={14} aria-hidden="true"/></AppLink>}
  </form>}
  {matches&&<div className="interview-batch-save-confirmed" role="status"><CheckCircle2 size={16} aria-hidden="true"/><span>{complete?t(`已保存到“${saved!.title}”。`,`Saved to “${saved!.title}”.`):t('保存已确认；题单内容后来有变化。','The save is confirmed; the list has since changed.')}</span>{saved&&<AppLink href={favoriteHref({query:'',state:'all',list:saved.id})}>{t('查看题单','View study list')}<ArrowRight size={14} aria-hidden="true"/></AppLink>}</div>}
 </section>;
}
