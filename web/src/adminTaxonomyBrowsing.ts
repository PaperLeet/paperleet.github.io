import type {Term} from './AdminParts';

export type TaxonomyBrowse={query:string;status:'all'|'active'|'paused';limit:number};
export const taxonomyPageSize=20;
export function taxonomyBrowse(value?:Partial<TaxonomyBrowse>):TaxonomyBrowse{
 const limit=value?.limit;
 return {query:typeof value?.query==='string'?value.query.slice(0,100):'',status:value?.status==='active'||value?.status==='paused'?value.status:'all',limit:typeof limit==='number'&&Number.isSafeInteger(limit)&&limit>=20?Math.min(200,limit):20};
}
const normalize=(value:string)=>value.normalize('NFKC').toLowerCase();
/** Local metadata browsing only. A match never changes write permissions or a form draft. */
export function matchingAdminTerms(rows:readonly Term[],view:TaxonomyBrowse){
 const words=normalize(view.query).trim().split(/\s+/).filter(Boolean);
 return rows.filter(row=>{if(view.status!=='all'&&row.enabled!==(view.status==='active'))return false;const text=normalize([row.id,row.zh,row.en,row.description].join(' '));return words.every(word=>text.includes(word));});
}
export function validAdminTaxonomy(value:{boards:Term[];tags:Term[]}){
 return ['boards','tags'].every(key=>{
  const rows=value?.[key as 'boards'];
  return Array.isArray(rows)&&rows.length<=(key==='boards'?50:200)&&new Set(rows.map(row=>row?.id)).size===rows.length&&rows.every(row=>row&&typeof row.id==='string'&&/^[a-z][a-z0-9_-]{0,39}$/.test(row.id)&&['zh','en','description'].every(field=>typeof row[field as 'zh']==='string')&&Number.isSafeInteger(row.revision)&&row.revision>=0&&typeof row.enabled==='boolean'&&Number.isSafeInteger(row.position)&&row.position>=0&&row.position<=9999&&Number.isSafeInteger(row.posts)&&row.posts>=0);
 });
}
