import type {InterviewSummary} from './interviewModels';
export type LibraryCompanyData={scope:string;phase:'loading'|'ready'|'error'|'unavailable';company:string;days:string;ids:ReadonlySet<string>;summary?:InterviewSummary;error?:string;missing?:number};
export type LibraryCompanyScope={scope:string;data?:LibraryCompanyData;retry:()=>void;reload:()=>void};
export type LibraryCompanyStatusProps={company:string;days:string;returnTo:string}&LibraryCompanyScope;
export type LibraryCompanyControlsProps={company:string;days:string;onChange:(company:string,days:string)=>void};
