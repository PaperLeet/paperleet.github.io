import {ArrowUpRight,BookOpen,CheckCircle2} from 'lucide-react';
import {useI18n} from './i18n';

export type InterviewDraftStep='title'|'body'|'record'|'ready'|'retry'|'busy'|'anonymous'|'account'|'account-loading';
type Copy={title:[string,string];detail:[string,string];action?:[string,string]};
const copy:Record<InterviewDraftStep,Copy>={
 title:{title:['给这次经历起个标题','Give your experience a title'],detail:['概括岗位、轮次或这次面试的重点，方便大家找到你的分享。','Summarize the role, round or main topic so readers can find your experience.'],action:['填写标题','Add a title']},
 body:{title:['补充你的面试经历','Add your interview experience'],detail:['写下实际的流程、思路或复盘；只有提纲时还不能发布。','Describe your actual process, approach or reflection. An empty outline is not ready to publish.'],action:['继续写正文','Continue writing']},
 record:{title:['有一条问题记录尚未插入','A question record is not in your report yet'],detail:['返回记录面板，插入正文或清空记录后再发布。输入会继续保存在草稿中。','Return to the record panel to insert or clear it before publishing. Your inputs stay in the draft.'],action:['处理问题记录','Review question record']},
 ready:{title:['标题和正文已填写','Your title and report are filled in'],detail:['可以继续编辑，或使用正文上方的预览检查排版。','Keep editing, or check the formatting with Preview above the report.'],action:['返回正文','Back to report']},
 retry:{title:['内容与待确认发布一致','Your draft matches the pending publication'],detail:['重试会继续确认同一次发布；返回正文修改会保留当前草稿。','Retry to confirm the same publication, or return to editing your draft.'],action:['返回正文','Back to report']},
 anonymous:{title:['匿名发布目前不可用','Anonymous publishing is unavailable'],detail:['草稿及匿名选项已保留。你可以查看发布身份，自主选择是否改用昵称发布。','Your draft and anonymous choice are kept. Review your posting identity to choose whether to use your public name.'],action:['查看发布身份','Review posting identity']},
 account:{title:['当前账号暂时无法发布','Publishing is unavailable for this account'],detail:['当前草稿仍可编辑和保留，账号状态就绪后可继续发布。','You can keep editing your draft and publish when your account is ready.']},
 'account-loading':{title:['正在读取账号状态…','Checking your account status…'],detail:['草稿仍保留，暂时无法确认是否可以发布。','Your draft is kept while publishing availability is being checked.']},
 busy:{title:['正在确认发布结果…','Confirming your publication…'],detail:['当前草稿会继续保留。','Your current draft is kept.']},
};

export default function InterviewDraftStatus({step,id,controls,disabled,onNavigate}:{step:InterviewDraftStep;id:string;controls:string;disabled:boolean;onNavigate:(opener:HTMLButtonElement)=>void}){
 const {t}=useI18n(),text=copy[step],Icon=step==='ready'||step==='retry'?CheckCircle2:BookOpen;
 return <aside className="interview-draft-status" data-step={step} aria-label={t('面经草稿状态','Interview draft status')}>
  <span className="interview-draft-status-mark" aria-hidden="true">{step==='busy'||step==='account-loading'?<span className="spinner"/>:<Icon size={20}/>}</span>
  <p id={id} role="status" aria-atomic="true"><strong>{t(...text.title)}</strong>{' '}<span>{t(...text.detail)}</span></p>
  {text.action&&<button type="button" aria-controls={controls} disabled={disabled} onClick={event=>onNavigate(event.currentTarget)}>{t(...text.action)}<ArrowUpRight size={16} aria-hidden="true"/></button>}
 </aside>;
}
