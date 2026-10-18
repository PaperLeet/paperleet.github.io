import {PRACTICE_GROUP_LIMIT} from './interviewBatchSharing';

export type BudgetMode='remaining'|'all';
export type BudgetItem={id:string;number:number;minutes?:number;state:'done'|'started'|'todo'|'unknown'};
export type BudgetInput={scope:string;owner:string;selected:readonly number[];items:readonly BudgetItem[];candidates:readonly number[];ready:boolean;pending:boolean};
export type BudgetReason='budget'|'pending'|'unavailable'|'estimate'|'over'|'full'|'progress'|'empty'|'fit';
export type BudgetPlan={fingerprint:string;scope:string;owner:string;budget:number|undefined;mode:BudgetMode;selected:number[];additions:BudgetItem[];selectedMinutes:number;minutes:number;unknownSelected:number;unknownCandidates:number;unavailableSelected:number;skippedPassed:number;skippedLong:number;minimum?:number;reason?:BudgetReason};
export function readPracticeBudget(raw:string){const value=raw.normalize('NFKC').trim();return /^\d{1,3}$/.test(value)&&+value>=1&&+value<=240?+value:undefined;}
const estimate=(value:unknown):value is number=>typeof value==='number'&&Number.isSafeInteger(value)&&value>0&&value<=1440;
const validNumber=(value:number)=>Number.isSafeInteger(value)&&value>0&&value<=99999;

/** Greedy in the existing evidence/curriculum order, never a frequency or completion claim. */
export function interviewBudgetPlan(input:BudgetInput,budget:number|undefined,mode:BudgetMode):BudgetPlan{
 const counts=new Map<number,number>(),ids=new Map<string,number>();
 for(const item of input.items){counts.set(item.number,(counts.get(item.number)||0)+1);ids.set(item.id,(ids.get(item.id)||0)+1);}
 const items=input.items.filter(item=>validNumber(item.number)&&/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(item.id)&&counts.get(item.number)===1&&ids.get(item.id)===1);
 const byNumber=new Map(items.map(item=>[item.number,item])),selected=[...input.selected];
 const normalized=input.items.map(item=>[item.id,item.number,estimate(item.minutes)?item.minutes:null,item.state]);
 const fingerprint=JSON.stringify([input.scope,input.owner,selected,normalized,input.candidates,input.ready,input.pending,budget??null,mode]);
 const unavailableSelected=selected.filter(number=>!byNumber.has(number)).length;
 const unknownSelected=selected.filter(number=>byNumber.has(number)&&!estimate(byNumber.get(number)!.minutes)).length;
 const selectedMinutes=selected.reduce((sum,number)=>sum+(estimate(byNumber.get(number)?.minutes)?byNumber.get(number)!.minutes!:0),0);
 const result:BudgetPlan={fingerprint,scope:input.scope,owner:input.owner,budget,mode,selected,additions:[],selectedMinutes,minutes:selectedMinutes,unknownSelected,unknownCandidates:0,unavailableSelected,skippedPassed:0,skippedLong:0};
 if(!Number.isSafeInteger(budget)||!budget||budget<1||budget>240){result.reason='budget';return result;}
 if(input.pending){result.reason='pending';return result;}
 if(selected.length>PRACTICE_GROUP_LIMIT||new Set(selected).size!==selected.length||unavailableSelected){result.reason='unavailable';return result;}
 if(unknownSelected){result.reason='estimate';return result;}
 if(selectedMinutes>budget){result.reason='over';return result;}
 if(selected.length===PRACTICE_GROUP_LIMIT){result.reason='full';return result;}
 if(mode==='remaining'&&!input.ready){result.reason='progress';return result;}
 const candidates=[...new Set(input.candidates)].flatMap(number=>!selected.includes(number)&&byNumber.has(number)?[byNumber.get(number)!]:[]);
 let eligible=0;
 for(const item of candidates){
  if(mode==='remaining'&&item.state==='done'){result.skippedPassed++;continue;}
  if(mode==='remaining'&&item.state==='unknown'){result.reason='progress';result.additions=[];result.minutes=selectedMinutes;return result;}
  eligible++;
  if(!estimate(item.minutes)){result.unknownCandidates++;continue;}
  result.minimum=Math.min(result.minimum??Infinity,selectedMinutes+item.minutes);
  if(item.minutes>budget-result.minutes){result.skippedLong++;continue;}
  if(result.additions.length>=PRACTICE_GROUP_LIMIT-selected.length)continue;
  result.additions.push(item);result.minutes+=item.minutes;
 }
 if(!result.additions.length)result.reason=eligible?'fit':'empty';
 return result;
}
