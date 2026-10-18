import {communityCategories,isCommunityCategory} from './communityCategories';
export type CommunityTerm={id:string;zh:string;en:string;description:string;enabled:boolean;position:number;revision:number};
export type CommunityTaxonomy={revision:number;boards:CommunityTerm[];tags:CommunityTerm[]};
export const fallbackTaxonomy:CommunityTaxonomy={revision:0,boards:communityCategories.map((value,position)=>({...value,description:'',enabled:true,position,revision:0})),tags:[]};
export function readCommunityTaxonomy(value:CommunityTaxonomy){
 const valid=(item:CommunityTerm)=>item&&isCommunityCategory(item.id)&&typeof item.zh==='string'&&!!item.zh&&item.zh.length<=40&&typeof item.en==='string'&&!!item.en&&item.en.length<=60&&typeof item.description==='string'&&item.description.length<=240&&typeof item.enabled==='boolean'&&Number.isSafeInteger(item.position)&&item.position>=0&&item.position<=9999&&Number.isSafeInteger(item.revision)&&item.revision>=0;
 if(!value||!Number.isSafeInteger(value.revision)||value.revision<1||!Array.isArray(value.boards)||!value.boards.length||value.boards.length>50||!Array.isArray(value.tags)||value.tags.length>200||[value.boards,value.tags].some(rows=>rows.some(row=>!valid(row))||new Set(rows.map(row=>row.id)).size!==rows.length))throw Error('Community boards and tags could not be confirmed.');
 return value;
}
