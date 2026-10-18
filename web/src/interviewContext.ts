export const INTERVIEW_CONTEXT_KEYS=['location','month_from','month_to'] as const;
export type InterviewContext={location:string;month_from:string;month_to:string};
export const emptyInterviewContext=():InterviewContext=>({location:'',month_from:'',month_to:''});
export function hasInterviewContext(value:InterviewContext){return !!(value.location||value.month_from||value.month_to);}
const month=(value:string)=>!value||/^(19|20)\d{2}-(0[1-9]|1[0-2])$/.test(value);
export function interviewContextError(value:InterviewContext):'location'|'month'|'order'|undefined{
 if(Array.from(value.location).length>80||/[\u0000-\u001f\u007f]/.test(value.location))return 'location';
 if(!month(value.month_from)||!month(value.month_to))return 'month';
 if(value.month_from&&value.month_to&&value.month_from>value.month_to)return 'order';
}
export function matchesInterviewContext(raw:unknown,expected:InterviewContext=emptyInterviewContext()){
 if(raw===undefined)return !hasInterviewContext(expected);
 if(!raw||typeof raw!=='object'||Array.isArray(raw)||Object.keys(raw).length!==3||Object.keys(raw).some(key=>!(INTERVIEW_CONTEXT_KEYS as readonly string[]).includes(key)))return false;
 const value=raw as InterviewContext;
 if(INTERVIEW_CONTEXT_KEYS.some(key=>typeof value[key]!=='string')||interviewContextError(value))return false;
 return INTERVIEW_CONTEXT_KEYS.every(key=>value[key]===expected[key]);
}
export function appendInterviewContext(params:URLSearchParams,value:InterviewContext){for(const key of INTERVIEW_CONTEXT_KEYS)if(value[key])params.set(key,value[key]);}
const locationText=(value:string)=>value.normalize('NFKC').trim().toLowerCase();
export function matchesInterviewPostContext(details:{location:string;interview_month:string},expected:InterviewContext){
 return (!expected.location||locationText(details.location).includes(locationText(expected.location)))&&(!expected.month_from||!!details.interview_month&&details.interview_month>=expected.month_from)&&(!expected.month_to||!!details.interview_month&&details.interview_month<=expected.month_to);
}
