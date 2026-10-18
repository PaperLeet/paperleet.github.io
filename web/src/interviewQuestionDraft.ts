export type InterviewQuestionScratch={module_id:string;prompt:string;approach:string;followups:string};
export const QUESTION_FIELD_LIMIT=3000;
export const emptyInterviewQuestion=():InterviewQuestionScratch=>({module_id:'',prompt:'',approach:'',followups:''});
export function readInterviewQuestion(value:unknown):InterviewQuestionScratch|undefined{
 if(!value||typeof value!=='object'||Array.isArray(value))return;
 const row=value as Record<string,unknown>;
 if(Object.keys(row).sort().join(',')!=='approach,followups,module_id,prompt'||typeof row.module_id!=='string'||row.module_id&&!/^[a-z0-9_-]{1,80}\/[a-z0-9_-]{1,80}$/.test(row.module_id))return;
 for(const field of ['prompt','approach','followups']as const)if(typeof row[field]!=='string'||row[field].length>QUESTION_FIELD_LIMIT||/[\uD800-\uDFFF]/u.test(row[field]))return;
 return {module_id:row.module_id,prompt:row.prompt as string,approach:row.approach as string,followups:row.followups as string};
}
export const hasInterviewQuestionText=(value?:InterviewQuestionScratch)=>!!value&&[value.prompt,value.approach,value.followups].some(text=>!!text.trim());
export const interviewQuestionSignature=(value?:InterviewQuestionScratch)=>value?JSON.stringify([value.module_id,value.prompt,value.approach,value.followups]):'';

export const interviewQuestionText=(value?:InterviewQuestionScratch)=>value?[value.prompt,value.approach,value.followups].filter(text=>!!text.trim()).join('\n\n'):'';
