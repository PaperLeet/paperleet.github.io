import {RefreshCw} from 'lucide-react';
import {deferredFeature} from './deferredFeature';
import {useI18n} from './i18n';
type Props=Parameters<typeof import('./ProblemFavorites').ProblemFavoriteButton>[0];
function Retry({retry,problemName,canAct}:Props&{retry:()=>void}){
 const {t}=useI18n(),name=problemName||t('本题','this exercise');
 return <button type="button" className="library-exercise-favorite-retry" aria-label={t(`重新加载${name}的收藏按钮`,`Reload save button for ${name}`)} title={t('重新加载收藏按钮','Reload save button')} onClick={event=>{
  if(canAct?.()===false||event.currentTarget.closest('[hidden],[aria-hidden="true"]'))return;
  event.currentTarget.closest('.library-exercise-row-actions')?.querySelector<HTMLButtonElement>('.library-exercise-company-trigger')?.focus({preventScroll:true});retry();
 }}><RefreshCw size={16} aria-hidden="true"/></button>;
}
/** One shared owner snapshot serves all row buttons; no report source is loaded here. */
export default deferredFeature<Props>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoriteButton})),['题目收藏','saved problems'],{loading:()=> <span className="library-exercise-favorite-placeholder" aria-hidden="true"/>,failure:(props,retry)=><Retry {...props} retry={retry}/>});
