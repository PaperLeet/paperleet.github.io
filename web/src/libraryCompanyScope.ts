import type {InterviewSummary} from './interviewModels';
import type {Problem} from './types';
export type LibraryCompanyData={scope:string;phase:'loading'|'ready'|'error'|'unavailable';company:string;days:string;ids:ReadonlySet<string>;summary?:InterviewSummary;error?:string;missing?:number};
export type LibraryCompanyScope={scope:string;data?:LibraryCompanyData;retry:()=>void;reload:()=>void};
export type LibraryCompanyStatusProps={company:string;days:string;returnTo:string}&LibraryCompanyScope;
export type LibraryCompanyControlsProps={company:string;days:string;onChange:(company:string,days:string)=>void};
export type LibraryCompanyRowToolsProps={problem:Problem;route:string;returnTo:string;companyScope:LibraryCompanyScope};
export type LibraryCompanyGeneralRowProps=Omit<LibraryCompanyRowToolsProps,'companyScope'>&{companyScope?:undefined};
export type LibraryCompanyToolsProps=LibraryCompanyGeneralRowProps|LibraryCompanyControlsProps|LibraryCompanyRowToolsProps|LibraryCompanyStatusProps;
