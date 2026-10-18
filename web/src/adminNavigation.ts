/** Local administrator views only. Never include review notes or captured commands in a URL. */
export type AdminSection = '' | 'boards' | 'tags' | 'reports' | 'users' | 'papers' | 'audit';
export type AdminView = {section:AdminSection; page:number; q?:string; status?:string; review?:string; kind?:string; item?:string; create?:boolean};
export const adminOptions:Record<string,Record<string,string[]>>={
 users:{status:['all','active','disabled','waitlisted']},
 papers:{status:['pending_review','all','suitability','generation','validation','pool','ready','changes','rejected','published'],review:['all','pending','approved']},
 audit:{kind:['all','boards','tags','users','papers']},
};
export function readAdminRoute(route:string):AdminView|undefined {
 if(route.length>1600)return;
 const [path,query,...extra]=route.replace(/^#\/?/,'').split('?');
 const match=path.match(/^community\/admin(?:\/(boards|tags|reports|users|papers|audit))?$/);
 if(!match||extra.length)return;
 const section=(match[1]||'') as AdminSection,params=new URLSearchParams(query||'');
 try{encodeURIComponent(decodeURIComponent((query||'').replace(/\+/g,' ')));}catch{return;}
 const options=adminOptions[section]||{},allowed=[...Object.keys(options),...(['users','papers','audit'].includes(section)?['page']:[]),...(section==='users'?['q']:[]),...(['boards','tags','users','papers'].includes(section)?['item']:[]),...(['boards','tags'].includes(section)?['create']:[])];
 if([...params.keys()].some(key=>!allowed.includes(key)||params.getAll(key).length!==1))return;
 const page=params.get('page')??'1';if(!/^[1-9]\d{0,2}$/.test(page)||Number(page)>501)return;
 const result:AdminView={section,page:Number(page)};
 for(const [key,values]of Object.entries(options)){const value=params.get(key)??values[0];if(!values.includes(value))return;Object.assign(result,{[key]:value});}
 if(section==='users'){const raw=params.get('q')||'',q=raw.trim();if(Array.from(q).length>100||/[\x00-\x1f\x7f]/.test(raw))return;result.q=q;}
 if(params.has('item')){const item=params.get('item')!;if(!(['users','papers'].includes(section)?/^[a-f0-9]{32}$/:/^[a-z][a-z0-9_-]{0,39}$/).test(item))return;result.item=item;}
 if(params.has('create')){if(params.get('create')!=='1'||result.item)return;result.create=true;}
 return result;
}
