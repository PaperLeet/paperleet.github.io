export const MAX_INTERVIEW_PAGE=5000;
export type InterviewStudyState='all'|'todo'|'started'|'done';
export type InterviewCompanyOrder='catalog'|'contributors'|'reports'|'name';
export type InterviewFilters={company:string;role:string;stage:string;q:string;days:'all'|'30'|'90'|'365';sort:'recent'|'helpful';page:number;view:'reports'|'questions'|'companies';track:string;saved:boolean;concept:string;study:InterviewStudyState;queue:string;company_order:InterviewCompanyOrder;outcome:string;compare?:string;comparison?:'shared'|'different'|'all'};
export const emptyInterviewFilters=():InterviewFilters=>({company:'',role:'',stage:'',q:'',days:'365',sort:'recent',page:1,view:'reports',track:'',saved:false,concept:'',study:'all',queue:'',company_order:'catalog',outcome:''});
export function readInterviewRoute(route:string):InterviewFilters|undefined {
  if(route.length>8192)return;
  const value=route.replace(/^#\/?/,'');const [path,search='']=value.split('?');
  if(path!=='interviews'||value.split('?').length>2)return;
  try{decodeURIComponent(search.replaceAll('+','%20'));}catch{return;}
  const params=new URLSearchParams(search),filters=emptyInterviewFilters();
  for(const [key,value] of params){
    if(params.getAll(key).length!==1||!['company','role','stage','q','days','sort','page','view','track','saved','concept','study','queue','company_order','outcome','compare','comparison'].includes(key))return;
    if(key==='compare'){if(!/^[a-z][a-z0-9-]{0,39},[a-z][a-z0-9-]{0,39}$/.test(value)||new Set(value.split(',')).size!==2)return;filters.compare=value;}
    else if(key==='comparison'){if(!['shared','different','all'].includes(value))return;filters.comparison=value as InterviewFilters['comparison'];}
    else if(key==='saved'){if(!['0','1'].includes(value))return;filters.saved=value==='1';}
    else if(key==='concept'){if(!/^[a-z0-9_-]{1,80}\/[a-z0-9_-]{1,80}$/.test(value))return;filters.concept=value;}
    // Keep the routing list aligned with the catalog without importing its deferred metadata here.
    else if(key==='track'){if(!['training-generation','llm','normalization','recommendation','training','representation','transformer','post-training','few-step-generation','world-models','representation-shape','policy-distillation','visual-quantization','feedback-distillation','advanced-architecture','off-policy-passk','generation-targets','low-rank-quantization','block-generation','inverse-diffusion'].includes(value))return;filters.track=value;}
    else if(key==='study'){if(!['all','todo','started','done'].includes(value))return;filters.study=value as InterviewStudyState;}
    else if(key==='outcome'){if(!['unknown','pending','offer','rejected','withdrew'].includes(value))return;filters.outcome=value;}
    else if(key==='company_order'){if(!['catalog','contributors','reports','name'].includes(value))return;filters.company_order=value as InterviewCompanyOrder;}
    else if(key==='queue'){if(!/^[1-9]\d{0,4}(?:,[1-9]\d{0,4}){0,79}$/.test(value)||new Set(value.split(',')).size!==value.split(',').length)return;filters.queue=value;}
    else if(key==='page'){if(!/^[1-9]\d{0,3}$/.test(value)||Number(value)>MAX_INTERVIEW_PAGE)return;filters.page=Number(value);}
    else if(key==='q'){if(Array.from(value).length>200||/[\u0000-\u001f\u007f]/.test(value))return;filters.q=value;}
    else if(key==='days'){if(!['all','30','90','365'].includes(value))return;filters.days=value as InterviewFilters['days'];}
    else if(key==='sort'){if(!['recent','helpful'].includes(value))return;filters.sort=value as InterviewFilters['sort'];}
    else if(key==='view'){if(!['reports','questions','companies'].includes(value))return;filters.view=value as InterviewFilters['view'];}
    else {if(!/^[a-z][a-z0-9-]{0,39}$/.test(value))return;filters[key as 'company'|'role'|'stage']=value;}
  }
  if((filters.compare||filters.comparison)&&(filters.view!=='companies'||!!filters.company||!filters.compare))return;
  return filters;
}
export function interviewHref(filters:InterviewFilters=emptyInterviewFilters()){
  const defaults=emptyInterviewFilters(),params=new URLSearchParams();
  for(const key of Object.keys(defaults) as (keyof InterviewFilters)[])if(filters[key]!==defaults[key]&&(key!=='study'||filters.view==='questions')&&(key!=='queue'||interviewBatchView(filters))&&(key!=='company_order'||filters.view==='companies'))params.set(key,key==='saved'?'1':String(filters[key]));
  if(filters.compare&&filters.view==='companies'&&!filters.company){params.set('compare',filters.compare);if(filters.comparison&&filters.comparison!=='shared')params.set('comparison',filters.comparison);}
  const href='#/interviews'+(params.size?'?'+params:'');
  if(!readInterviewRoute(href))throw Error('Invalid interview filters');
  return href;
}

/** Captured practice groups may return to a ranked list or an applied company comparison. */
export function interviewBatchView(filters:InterviewFilters){return filters.view==='questions'||filters.view==='companies'&&!!filters.compare&&!filters.company;}
