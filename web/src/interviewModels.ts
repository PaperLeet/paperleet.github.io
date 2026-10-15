import rawCatalog from '../../paperleet/interview_catalog.json';
import type {Post} from './communityDiscussion';

export const interviewCatalog=rawCatalog;
export type InterviewDetails={company:string;role:string;stage:string;outcome:string;interview_month:string;location:string;questions:string[]};
export type InterviewPost=Post&{interview:InterviewDetails};
export function interviewLabel(group:'companies'|'roles'|'stages'|'outcomes',id:string,en:boolean){
  const row=interviewCatalog[group].find(row=>row.id===id);
  return row?(en?row.name_en:row.name):id;
}
export function readInterviewDetails(raw:unknown,now=new Date()):InterviewDetails|undefined {
  if(!raw||typeof raw!=='object'||Array.isArray(raw))return;
  const value=raw as Record<string,unknown>;
  if(Object.keys(value).some(key=>!['company','role','stage','outcome','interview_month','location','questions'].includes(key)))return;
  for(const [key,group] of [['company','companies'],['role','roles'],['stage','stages'],['outcome','outcomes']] as const)
    if(typeof value[key]!=='string'||!interviewCatalog[group].some(row=>row.id===value[key]))return;
  if(typeof value.interview_month!=='string'||(value.interview_month&&(!/^(19|20)\d{2}-(0[1-9]|1[0-2])$/.test(value.interview_month)||value.interview_month>now.toISOString().slice(0,7))))return;
  if(typeof value.location!=='string'||value.location.length>80||/[\u0000-\u001f]/.test(value.location))return;
  if(!Array.isArray(value.questions)||value.questions.length>12||new Set(value.questions).size!==value.questions.length||value.questions.some(q=>typeof q!=='string'||!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(q)))return;
  return {company:value.company as string,role:value.role as string,stage:value.stage as string,outcome:value.outcome as string,interview_month:value.interview_month,location:value.location.trim(),questions:[...value.questions as string[]].sort()};
}
export type InterviewSummary={reports:number;contributors:number;days:string;basis:'published-community-reports';generated_at:string;companies:{company:string;reports:number;contributors:number}[];questions:{problem_id:string;reports:number;contributors:number;companies:{company:string;reports:number;contributors:number;latest:string;problem_id:string}[]}[]};
export type InterviewFeed={items:InterviewPost[];total:number;limit:number;offset:number;owner_id?:string|null};
const count=(value:unknown):value is number=>Number.isSafeInteger(value)&&Number(value)>=0;
const companyCount=(row:unknown)=>{
  if(!row||typeof row!=='object')return false;
  const value=row as Record<string,unknown>;
  return interviewCatalog.companies.some(company=>company.id===value.company)&&count(value.reports)&&count(value.contributors)&&value.contributors<=value.reports;
};
export function readInterviewFeed(raw:unknown):InterviewFeed|undefined {
  const value=raw as InterviewFeed;
  if(!value||!Array.isArray(value.items)||!count(value.total)||!count(value.offset)||!count(value.limit)||value.limit<1||value.limit>40||value.items.length>value.limit)return;
  if(value.owner_id!==undefined&&value.owner_id!==null&&typeof value.owner_id!=='string')return;
  if(value.items.some(post=>!post||!readInterviewDetails(post.interview)||!/^[a-f0-9]{32}$/.test(post.id)||post.category!=='career'||typeof post.title!=='string'||typeof post.body!=='string'||!post.author||typeof post.author.id!=='string'||typeof post.author.username!=='string'||typeof post.author.display_name!=='string'||typeof post.author.avatar_color!=='string'||typeof post.created_at!=='string'||!post.counts||![post.counts.likes,post.counts.replies,post.counts.reposts].every(count)))return;
  return value;
}
export function readInterviewSummary(raw:unknown):InterviewSummary|undefined {
  const value=raw as InterviewSummary;
  if(!value||value.basis!=='published-community-reports'||!count(value.reports)||!count(value.contributors)||value.contributors>value.reports||!Array.isArray(value.companies)||!Array.isArray(value.questions)||!value.companies.every(companyCount))return;
  if(value.questions.some(row=>!row||typeof row.problem_id!=='string'||!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(row.problem_id)||!count(row.reports)||!count(row.contributors)||row.contributors>row.reports||!Array.isArray(row.companies)||!row.companies.every(companyCount)))return;
  return value;
}

export {readInterviewRoute,interviewHref,emptyInterviewFilters,type InterviewFilters} from './interviewNavigation';
