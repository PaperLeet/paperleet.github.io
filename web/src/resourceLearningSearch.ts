import type {Problem} from './types';
import {resolveResourcePractice} from './resourcePractice';
import {foldResourceText,resourceBrowserTerms,resourceSearchText,type InterviewResource} from './interviewResourceBrowsing';

export type ResourceLearningField={key:string;kind:'exercise'|'note'|'question';text:string;lang?:'en'|'zh-CN';problem?:Problem};
/** Index only maintained learning text and uniquely resolved public exercise labels. */
export function resourceLearningFields(resource:string,problems:Problem[],en:boolean):ResourceLearningField[]{
 const practice=resolveResourcePractice(resource,problems);if(!practice)return [];
 const fields:ResourceLearningField[]=practice.modules.map(problem=>{
  const english=problem.translations?.en?.title||problem.title,names=[...new Set(en?[english,problem.title]:[problem.title,english])];
  return {key:'exercise-'+problem.id,kind:'exercise',text:[`${problem.number}. ${names[0]}`,...names.slice(1),problem.id].join('\n'),problem};
 });
 for(const language of en?['en','zh']as const:['zh','en']as const){
  const lang=language==='zh'?'zh-CN':'en';fields.push({key:'note-'+language,kind:'note',text:practice[language],lang});
  practice.questions?.forEach((question,index)=>fields.push({key:`question-${index}-${language}`,kind:'question',text:question[language],lang}));
 }
 return fields;
}
/** Keep editorial explanations outside official-source text; cover only missing metadata terms. */
export function resourceLearningContexts(resource:InterviewResource,query:string,fields:ResourceLearningField[]){
 const metadata=resourceSearchText(resource);let missing=[...new Set(resourceBrowserTerms(query).split(' ').filter(term=>!!term&&!metadata.includes(term)))];
 if(!missing.length)return [];
 return fields.filter(field=>{const text=foldResourceText(field.text),found=missing.filter(term=>text.includes(term));if(!found.length)return false;missing=missing.filter(term=>!found.includes(term));return true;});
}
