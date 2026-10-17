import {interviewBatchShareUrl} from './interviewBatchSharing';
import {readInterviewRoute} from './interviewNavigation';
import {interviewLabel} from './interviewModels';
import type {Problem} from './types';

/** Only ordered public catalog identifiers enter a private save command. */
export function captureInterviewStudyList(pageUrl:string,href:string,modules:Problem[]){
 try{
  interviewBatchShareUrl(pageUrl,href,modules);
  const ids=modules.map(module=>module.id);
  if(ids.some(id=>typeof id!=='string'||id.length>180||!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(id))||new Set(ids).size!==ids.length)return;
  return {href,ids:[...ids]};
 }catch{return;}
}
export function interviewStudyListTitle(href:string,en:boolean){
 const filters=readInterviewRoute(href),company=filters?.company?interviewLabel('companies',filters.company,en):'';
 const title=company?en?`${company} practice batch`:`${company}准备练习`:en?'Interview practice batch':'面试准备练习';
 return Array.from(title).slice(0,60).join('');
}
