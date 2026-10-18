import type {PublicUser} from './social';
export type HighlightPost={id:string;title:string;body_excerpt:string;category:string;created_at:string;author:PublicUser;content_version:string};
export type HighlightSlot={id:string;slot:number;revision:number;post_id:string|null;state:'empty'|'active'|'changed'|'unavailable';post:HighlightPost|null};
export type HighlightList={category:string;items:{slot:number;post:HighlightPost}[]};
export type AdminHighlightList={owner_id:string;category:string;slots:HighlightSlot[]};
const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const slug=(v:unknown):v is string=>typeof v==='string'&&/^[a-z][a-z0-9_-]{0,39}$/.test(v);
const id=(v:unknown):v is string=>typeof v==='string'&&/^[a-f0-9]{32}$/.test(v);
const text=(v:unknown,max:number)=>typeof v==='string'&&v.length<=max&&!/[\uD800-\uDFFF]/u.test(v);
export const highlightResource=(category:string,slot:number)=>(category?'board:'+category:'global')+':'+slot;
export function highlightPostId(value:string):string|undefined{
 const input=value.trim();if(/^[a-f0-9]{32}$/.test(input))return input;if(input.length>1800)return;
 try{
  const url=new URL(input,'https://paperleet.invalid/');
  if(!['http:','https:'].includes(url.protocol)||url.username||url.password)return;
  if(url.hash.startsWith('#/'))return url.hash.match(/^#\/community\/([a-f0-9]{32})(?:\?[^#\s]*)?$/)?.[1];
  return url.pathname.match(/(?:^|\/)community\/([a-f0-9]{32})\/?$/)?.[1];
 }catch{return;}
}
export function isHighlightPost(v:unknown,category:string):v is HighlightPost{
 if(!object(v)||!id(v.id)||!text(v.title,180)||!v.title||!text(v.body_excerpt,2400)||!slug(v.category)||!!category&&v.category!==category||typeof v.content_version!=='string'||!/^[a-f0-9]{64}$/.test(v.content_version)||typeof v.created_at!=='string'||v.created_at.length>40||!/^\d{4}-\d{2}-\d{2}T/.test(v.created_at)||!Number.isFinite(Date.parse(v.created_at))||!object(v.author))return false;
 const a=v.author;
 if(!text(a.id,100)||!a.id||!text(a.username,80)||!text(a.display_name,160)||typeof a.avatar_color!=='string'||!/^#[a-f0-9]{6}$/i.test(a.avatar_color)||typeof a.bio!=='string'||typeof a.created_at!=='string'||a.anonymous!==undefined&&typeof a.anonymous!=='boolean')return false;
 if(a.anonymous===true&&(a.id!=='anonymous:'+v.id||a.username!==''||a.display_name!=='Anonymous'||a.avatar_url!==null||a.bio!==''||a.created_at!==''||a.show_progress!==false||a.is_test_user!==false||a.country!==undefined))return false;
 if(a.affiliation!=null&&(!object(a.affiliation)||!text(a.affiliation.name,240)||!text(a.affiliation.domain,253)||!['institution','email_domain'].includes(String(a.affiliation.kind))||a.affiliation.name_en!==undefined&&!text(a.affiliation.name_en,240)))return false;
 if(a.institution!=null&&(!object(a.institution)||!text(a.institution.id,100)||!text(a.institution.name,240)||a.institution.name_en!==undefined&&!text(a.institution.name_en,240)))return false;
 return true;
}
export function readHighlights(value:unknown,category:string):HighlightList|undefined{
 if(!object(value)||value.category!==category||category!==''&&!slug(category)||!Array.isArray(value.items)||value.items.length>3)return;
 let previous=0;const seen=new Set<string>();
 for(const row of value.items){if(!object(row)||!Number.isInteger(row.slot)||Number(row.slot)<=previous||Number(row.slot)>3||!isHighlightPost(row.post,category)||seen.has(row.post.id))return;previous=Number(row.slot);seen.add(row.post.id);}
 return value as HighlightList;
}
export function isAdminHighlights(value:unknown,category:string):value is AdminHighlightList{
 if(!object(value)||typeof value.owner_id!=='string'||!value.owner_id||value.category!==category||category!==''&&!slug(category)||!Array.isArray(value.slots)||value.slots.length!==3)return false;
 return value.slots.every((row,index)=>object(row)&&row.slot===index+1&&row.id===highlightResource(category,index+1)&&Number.isSafeInteger(row.revision)&&Number(row.revision)>=0&&['empty','active','changed','unavailable'].includes(String(row.state))&&
  (row.state==='empty'?row.post_id===null&&row.post===null:id(row.post_id)&&(row.state==='unavailable'?row.post===null:isHighlightPost(row.post,category)&&row.post.id===row.post_id)));
}
