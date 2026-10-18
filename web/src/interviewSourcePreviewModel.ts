import {appendInterviewContext,hasInterviewContext} from './interviewContext';
import {libraryReturnHref,readLibraryRoute,libraryHref} from './libraryNavigation';
import {interviewCatalog,readInterviewFeed} from './interviewModels';
import {interviewHref,readInterviewRoute,emptyInterviewFilters,type InterviewFilters} from './interviewNavigation';

export const INTERVIEW_SOURCE_PREVIEW_LIMIT=3;
/** A library preview is public evidence for its exact company/publication cohort. */
export function libraryInterviewPreviewScope(href:string,sourceHref:string):string|undefined{
 const library=readLibraryRoute(href),source=readInterviewRoute(sourceHref);
 if(!library?.filters.company||!source?.concept||!interviewSourceRequest(sourceHref))return;
 const expected=interviewHref({...emptyInterviewFilters(),company:library.filters.company,days:(library.filters.company_days||'365') as InterviewFilters['days'],concept:source.concept});
 if(interviewHref(source)!==expected)return;
 return libraryHref(library.collectionId,library.filters);
}
export function interviewSourceHref(filters:InterviewFilters,concept:string){
  return interviewHref({...filters,view:'reports',page:1,q:'',concept,saved:false,queue:'',study:'all',company_order:'catalog'});
}

/** Ranking queries match concept metadata, while source previews use public report evidence. */
export function interviewSourceRequest(href:string,contextAvailable=false){
  const filters=readInterviewRoute(href);
  if(!filters||filters.view!=='reports'||filters.saved||filters.q||!filters.concept)return;
  if(!([['company','companies'],['role','roles'],['stage','stages'],['outcome','outcomes']] as const).every(([field,group])=>!filters[field]||interviewCatalog[group].some(row=>row.id===filters[field])))return;
  if(hasInterviewContext(filters)&&contextAvailable!==true)return;
  const params=new URLSearchParams({days:filters.days,sort:filters.sort,limit:String(INTERVIEW_SOURCE_PREVIEW_LIMIT),offset:'0',problem_id:filters.concept});
  for(const key of ['company','role','stage','outcome'] as const)if(filters[key])params.set(key,filters[key]);
  appendInterviewContext(params,filters);
  return 'community/interviews?'+params;
}

export function readInterviewSourcePreview(raw:unknown,href:string,contextAvailable=false){
  if(!interviewSourceRequest(href,contextAvailable))return;
  const filters=readInterviewRoute(href)!,feed=readInterviewFeed(raw,filters.outcome,filters);
  if(!feed||feed.limit!==INTERVIEW_SOURCE_PREVIEW_LIMIT||feed.offset!==0||feed.items.length>feed.total||new Set(feed.items.map(post=>post.id)).size!==feed.items.length)return;
  if(feed.items.some(post=>post.solution===true||!post.interview.questions.includes(filters.concept)||(['company','role','stage'] as const).some(key=>filters[key]&&post.interview[key]!==filters[key])))return;
  if(feed.items.some(post=>(post.anonymous===true||post.author.anonymous===true||post.author.id.startsWith('anonymous:'))&&(post.author.anonymous!==true||post.author.id!=='anonymous:'+post.id||post.author.username!=='')))return;
  return feed;
}

/** A local practice preview may carry only its known public module and existing library context. */
export function interviewPracticePreviewScope(href:string,concept:string,number?:number):string|undefined{
 if(href.length>8192||!href.startsWith('#/'))return;
 const raw=href.slice(2),query=raw.indexOf('?');
 try{
  const path=decodeURIComponent(query<0?raw:raw.slice(0,query));
  if(path!==concept&&!(Number.isSafeInteger(number)&&number!==undefined&&number>0&&path===String(number)))return;
  if(!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(concept))return;
  if(query>=0){const params=new URLSearchParams(raw.slice(query+1));if([...params.keys()].length!==1||!params.get('from')||!libraryReturnHref(href))return;}
  return href;
 }catch{return;}
}
