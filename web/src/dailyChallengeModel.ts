import config from '../../paperleet/daily_challenges.json';
export const dailySchedule=config;
export type DailySelection={date:string;problem_id:string;version:number;timezone:string;ends_at:string};
export type DailyEntry={day:string;problem_id:string;started_at:string;completed_at:string|null;rating_before:number|null;rating_after:number|null;delta:number|null};
export type DailySnapshot={challenge:DailySelection;server_time:string;personal:null|{user_id:string;rating:number;completed_days:number;entry:DailyEntry|null;history:DailyEntry[]}};
export const utcDay=(date=new Date())=>date.toISOString().slice(0,10);
export function challengeMonth(month:string) {
  if(!/^\d{4}-\d{2}$/.test(month))return [];
  const first=new Date(month+'-01T00:00:00Z');
  if(!Number.isFinite(first.getTime())||utcDay(first).slice(0,7)!==month)return [];
  const count=new Date(Date.UTC(first.getUTCFullYear(),first.getUTCMonth()+1,0)).getUTCDate();
  return Array.from({length:count},(_,index)=>`${month}-${String(index+1).padStart(2,'0')}`);
}
export function dailySelection(day:string):DailySelection|undefined {
  if(!/^\d{4}-\d{2}-\d{2}$/.test(day))return;
  const time=Date.parse(day+'T00:00:00Z');
  if(!Number.isFinite(time)||utcDay(new Date(time))!==day)return;
  const offset=Math.round((time-Date.parse(config.starts_on+'T00:00:00Z'))/86400000);
  if(offset<0)return;
  return {date:day,problem_id:config.modules[offset%config.modules.length],version:config.version,timezone:'UTC',ends_at:new Date(time+86400000).toISOString()};
}
export function dailyRoute(value:string):{date?:string}|undefined {
  const raw=value.replace(/^#\/?/,'');
  if(raw==='daily-challenge')return {};
  const match=/^daily-challenge\?date=(\d{4}-\d{2}-\d{2})$/.exec(raw);
  return match&&dailySelection(match[1])?{date:match[1]}:undefined;
}
export function readDailySnapshot(value:unknown,owner?:string):DailySnapshot|undefined {
  if(!value||typeof value!=='object')return;
  const row=value as DailySnapshot,selection=row.challenge;
  if(!selection||!dailySelection(selection.date)||typeof selection.problem_id!=='string'||!/^\w+\/\w+$/.test(selection.problem_id)||!Number.isInteger(selection.version)||selection.timezone!=='UTC'||!Number.isFinite(Date.parse(selection.ends_at))||!Number.isFinite(Date.parse(row.server_time)))return;
  if(row.personal!==null){
    const personal=row.personal;
    if(!owner||!personal||personal.user_id!==owner||!Number.isSafeInteger(personal.rating)||personal.rating<1000||!Number.isSafeInteger(personal.completed_days)||personal.completed_days<0||!Array.isArray(personal.history)||personal.history.length>90)return;
    const valid=(entry:DailyEntry)=>!!entry&&typeof entry.day==='string'&&!!dailySelection(entry.day)&&typeof entry.problem_id==='string'&&typeof entry.started_at==='string'&&(entry.completed_at===null||typeof entry.completed_at==='string')&&(entry.delta===null||Number.isSafeInteger(entry.delta)&&entry.delta>0&&entry.delta<=32);
    if(personal.history.some(entry=>!valid(entry))||personal.entry!==null&&(!valid(personal.entry)||personal.entry.day!==selection.date||personal.entry.problem_id!==selection.problem_id))return;
  }
  return row;
}
