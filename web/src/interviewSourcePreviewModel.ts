import {interviewCatalog,readInterviewFeed} from './interviewModels';
import {interviewHref,readInterviewRoute,type InterviewFilters} from './interviewNavigation';

export const INTERVIEW_SOURCE_PREVIEW_LIMIT=3;
export function interviewSourceHref(filters:InterviewFilters,concept:string){
  return interviewHref({...filters,view:'reports',page:1,q:'',concept,saved:false,queue:'',study:'all',company_order:'catalog'});
}

/** Ranking queries match concept metadata, while source previews use public report evidence. */
export function interviewSourceRequest(href:string){
  const filters=readInterviewRoute(href);
  if(!filters||filters.view!=='reports'||filters.saved||filters.q||!filters.concept)return;
  if(!([['company','companies'],['role','roles'],['stage','stages'],['outcome','outcomes']] as const).every(([field,group])=>!filters[field]||interviewCatalog[group].some(row=>row.id===filters[field])))return;
  const params=new URLSearchParams({days:filters.days,sort:filters.sort,limit:String(INTERVIEW_SOURCE_PREVIEW_LIMIT),offset:'0',problem_id:filters.concept});
  for(const key of ['company','role','stage','outcome'] as const)if(filters[key])params.set(key,filters[key]);
  return 'community/interviews?'+params;
}

export function readInterviewSourcePreview(raw:unknown,href:string){
  if(!interviewSourceRequest(href))return;
  const filters=readInterviewRoute(href)!,feed=readInterviewFeed(raw,filters.outcome);
  if(!feed||feed.limit!==INTERVIEW_SOURCE_PREVIEW_LIMIT||feed.offset!==0||feed.items.length>feed.total||new Set(feed.items.map(post=>post.id)).size!==feed.items.length)return;
  if(feed.items.some(post=>post.solution===true||!post.interview.questions.includes(filters.concept)||(['company','role','stage'] as const).some(key=>filters[key]&&post.interview[key]!==filters[key])))return;
  if(feed.items.some(post=>(post.anonymous===true||post.author.anonymous===true||post.author.id.startsWith('anonymous:'))&&(post.author.anonymous!==true||post.author.id!=='anonymous:'+post.id||post.author.username!=='')))return;
  return feed;
}
