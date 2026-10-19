/** Local administrator views only. Review notes and captured commands never enter URLs. */
export type AdminSection = '' | 'boards' | 'tags' | 'reports' | 'users' | 'papers' | 'audit' | 'highlights';
export type AdminView = {section:AdminSection; page:number; q?:string; status?:string; review?:string; kind?:string; state?:string; minimum?:string; item?:string; create?:boolean; board?:string};
export const adminOptions:Record<string,Record<string,string[]>>={
 users:{status:['all','active','disabled','waitlisted']},
 papers:{status:['pending_review','all','suitability','generation','validation','pool','ready','changes','rejected','published'],review:['all','pending','approved']},
 reports:{state:['pending','resolved','all'],minimum:['1','3','5','10']},
 audit:{kind:['all','boards','tags','users','papers','highlights']},
};
export function readAdminRoute(route:string):AdminView|undefined {
 if(route.length>1600)return;
 const [path,query='',extra]=route.replace(/^#\/?/,'').split('?');
 const match=path.match(/^community\/admin(?:\/(boards|tags|reports|users|papers|audit|highlights))?$/);
 if(!match||extra!==undefined)return;
 try{encodeURIComponent(decodeURIComponent(query));}catch{return;}
 const section=(match[1]||'') as AdminSection,params=new URLSearchParams(query),options=adminOptions[section]||{},taxonomy=section==='boards'||section==='tags',resource=section==='users'||section==='papers';
 const result:AdminView={section,page:1};
 for(const key in options)result[key as 'status']=options[key][0];
 if(section==='users')result.q='';
 for(const [key,value]of params){
  if(params.getAll(key).length!==1)return;
  if(Object.hasOwn(options,key)){if(!options[key].includes(value))return;result[key as 'status']=value;}
  else if(key==='page'&&(resource||section==='audit'||section==='reports')){if(!/^[1-9]\d{0,2}$/.test(value)||+value>501)return;result.page=+value;}
  else if(key==='q'&&section==='users'){const q=value.trim();if([...q].length>100||/[\x00-\x1f\x7f]/.test(value))return;result.q=q;}
  else if(key==='item'&&(resource||taxonomy||section==='reports')){if(!(resource||section==='reports'?/^[a-f0-9]{32}$/:/^[a-z][a-z0-9_-]{0,39}$/).test(value))return;result.item=value;}
  else if(key==='board'&&section==='highlights'){if(!/^[a-z][a-z0-9_-]{0,39}$/.test(value))return;result.board=value;}
  else if(key==='create'&&taxonomy&&value==='1')result.create=true;
  else return;
 }
 if(result.item&&result.create)return;
 return result;
}
