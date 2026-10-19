import {useRef} from 'react';
import {RefreshCw} from 'lucide-react';
import {deferredFeature} from './deferredFeature';
import {appLocation} from './appLocation';
import {hostedAppAvailable} from './deployment';
import {useAccount} from './social';
import {useI18n} from './i18n';
import {interviewHref,readInterviewRoute} from './interviewNavigation';
import './interview-concept-favorite.css';

type ButtonProps=Parameters<typeof import('./ProblemFavorites').ProblemFavoriteButton>[0];
function Retry({retry,problemName,canAct}:ButtonProps&{retry:()=>void}){
 const {t}=useI18n();
 return <button type="button" className="interview-concept-save-retry" aria-label={t(`重新加载${problemName}的收藏按钮`,`Reload save button for ${problemName}`)} title={t('重新加载收藏按钮','Reload save button')} onClick={event=>{
  if(canAct?.()===false)return;
  event.currentTarget.closest('.interview-ranking-row')?.querySelector<HTMLAnchorElement>('.interview-ranking-title>a')?.focus({preventScroll:true});retry();
 }}><RefreshCw size={16} aria-hidden="true"/></button>;
}
const Save=deferredFeature<ButtonProps>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoriteButton})),['题目收藏','saved problems'],{loading:()=> <span className="interview-concept-save-placeholder" aria-hidden="true"/>,failure:(props,retry)=><Retry {...props} retry={retry}/>});

/** Saving a concept never changes its evidence rank, practice batch or source view. */
export default function InterviewConceptFavorite({problemId,problemName,returnTo,owner}:{problemId:string;problemName:string;returnTo:string;owner:string}){
 const {user,loading,capabilities}=useAccount(),root=useRef<HTMLDivElement>(null);
 const accountOwner=user?.id||'guest',live=useRef({owner:accountOwner,loading});live.current={owner:accountOwner,loading};
 if(!hostedAppAvailable||loading||capabilities?.problem_bookmarks!==true||owner!==accountOwner)return null;
 const canAct=()=>{
  const now=readInterviewRoute(appLocation.hash);
  return !live.current.loading&&live.current.owner===owner&&!!root.current?.isConnected&&!root.current.closest('[hidden],[aria-hidden="true"]')&&!!now&&now.view==='questions'&&interviewHref(now)===returnTo;
 };
 return <div ref={root} className="interview-concept-favorite" data-concept-save={problemId}><Save key={JSON.stringify([owner,returnTo,problemId])} problemId={problemId} problemName={problemName} returnTo={returnTo} showStudyLists={false} canAct={canAct}/></div>;
}
