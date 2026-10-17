import {interviewBatchView,interviewHref,readInterviewRoute,type InterviewFilters} from './interviewNavigation';
import {interviewViewShareUrl} from './InterviewViewLink';
import {interviewLabel} from './interviewModels';
import {withLibraryContext} from './libraryNavigation';
import type {Problem} from './types';

export const PRACTICE_GROUP_LIMIT=6;
export function interviewBatchShareHref(filters:InterviewFilters,modules:Problem[]):string {
  const numbers=modules.map(module=>module.number);
  if(!numbers.length||numbers.length>PRACTICE_GROUP_LIMIT||new Set(numbers).size!==numbers.length||numbers.some(number=>!Number.isSafeInteger(number)||number<1||number>99999))throw Error('Invalid practice batch');
  return interviewHref({...filters,view:interviewBatchView(filters)?filters.view:'questions',q:'',saved:false,study:'all',page:1,queue:numbers.join(',')});
}
const punctuation=new Set('\\`*_{}[]()<>#+-.!|~'.split(''));
const markdownText=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ')).map(character=>punctuation.has(character)?'\\'+character:character).join('');
const validMinutes=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value)&&value>0;

export function interviewBatchShareUrl(pageUrl:string,href:string,modules:Problem[]):string {
  const filters=readInterviewRoute(href);
  if(!filters||interviewBatchShareHref(filters,modules)!==interviewHref(filters))throw Error('Incomplete or mismatched practice batch');
  return interviewViewShareUrl(pageUrl,href);
}

/** Deliberately excludes workspace progress, report/source text, search terms and account identity. */
export function interviewBatchChecklist(pageUrl:string,href:string,modules:Problem[],en:boolean):string {
  const url=interviewBatchShareUrl(pageUrl,href,modules),page=new URL(url),filters=readInterviewRoute(href)!;
  const scope=[filters.company?interviewLabel('companies',filters.company,en):filters.compare?filters.compare.split(',').map(company=>interviewLabel('companies',company,en)).join(' / '):'',filters.role&&interviewLabel('roles',filters.role,en),filters.stage&&interviewLabel('stages',filters.stage,en),filters.outcome&&interviewLabel('outcomes',filters.outcome,en)].filter(Boolean).join(' · ');
  const times=modules.map(module=>module.estimated_minutes),total=times.every(validMinutes)?times.reduce<number>((sum,value)=>sum+value!,0):undefined;
  const lines=[en?'# PaperLeet practice checklist':'# PaperLeet 自选准备清单',''];
  if(scope)lines.push((en?'Preparation scope: ':'准备范围：')+markdownText(scope),'');
  if(total!==undefined&&Number.isFinite(total))lines.push(en?`Estimated total: about ${total} min`:`预计用时：整组约 ${total} 分钟`,'');
  for(const module of modules) {
    const title=(en?module.translations?.en?.title||module.title:module.title)||String(module.number);
    const target=page.origin+page.pathname+withLibraryContext(module.number,href);
    lines.push(`- [ ] [${markdownText(title)}](<${target}>)`+(validMinutes(module.estimated_minutes)?en?` · about ${module.estimated_minutes} min`:` · 约 ${module.estimated_minutes} 分钟`:''));
  }
  lines.push('',en?`[Open this practice batch](<${url}>)`:`[打开这组练习](<${url}>)`,'',en?'A self-selected study plan, not official company questions or interview frequencies. Progress uses the viewer’s own workspace.':'这是自选准备计划，不代表公司的官方考题或考频。完成进度以打开者自己的工作区为准。');
  return lines.join('\n');
}
