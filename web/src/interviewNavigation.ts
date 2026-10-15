export type InterviewFilters={company:string;role:string;stage:string;q:string;days:'all'|'30'|'90'|'365';sort:'recent'|'helpful';page:number;view:'reports'|'questions'|'companies';track:string};
export const emptyInterviewFilters=():InterviewFilters=>({company:'',role:'',stage:'',q:'',days:'365',sort:'recent',page:1,view:'reports',track:''});
export function readInterviewRoute(route:string):InterviewFilters|undefined {
  if(route.length>8192)return;
  const value=route.replace(/^#\/?/,'');const [path,search='']=value.split('?');
  if(path!=='interviews'||value.split('?').length>2)return;
  try{decodeURIComponent(search.replaceAll('+','%20'));}catch{return;}
  const params=new URLSearchParams(search),filters=emptyInterviewFilters();
  for(const [key,value] of params){
    if(params.getAll(key).length!==1||!['company','role','stage','q','days','sort','page','view','track'].includes(key))return;
    if(key==='track'){if(!['llm','recommendation','training'].includes(value))return;filters.track=value;}
    else if(key==='page'){if(!/^[1-9]\d{0,3}$/.test(value)||Number(value)>5000)return;filters.page=Number(value);}
    else if(key==='q'){if(value.length>200||/[\u0000-\u001f\u007f]/.test(value))return;filters.q=value;}
    else if(key==='days'){if(!['all','30','90','365'].includes(value))return;filters.days=value as InterviewFilters['days'];}
    else if(key==='sort'){if(!['recent','helpful'].includes(value))return;filters.sort=value as InterviewFilters['sort'];}
    else if(key==='view'){if(!['reports','questions','companies'].includes(value))return;filters.view=value as InterviewFilters['view'];}
    else {if(!/^[a-z][a-z0-9-]{0,39}$/.test(value))return;filters[key as 'company'|'role'|'stage']=value;}
  }
  return filters;
}
export function interviewHref(filters:InterviewFilters=emptyInterviewFilters()){
  const defaults=emptyInterviewFilters(),params=new URLSearchParams();
  for(const key of Object.keys(defaults) as (keyof InterviewFilters)[])if(filters[key]!==defaults[key])params.set(key,String(filters[key]));
  const href='#/interviews'+(params.size?'?'+params:'');
  if(!readInterviewRoute(href))throw Error('Invalid interview filters');
  return href;
}
