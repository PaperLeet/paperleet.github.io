import { parseReport } from './reports';
import { learningOwner, learningKey, mergeBrowserLearning, type BrowserLearning } from './browserLearning';
import type { Problem, Report } from './types';

function browserReport(value: unknown, id: string) {
  try {const report = parseReport(value); return report.problem === id && report.runtime === 'browser-numpy-wasm' ? report : undefined;}
  catch {return undefined;}
}
export function parseBrowserLearning(value: unknown, owner: string): BrowserLearning | undefined {
  if (!value || typeof value !== 'object' || !learningOwner(owner)) return;
  const row = value as BrowserLearning;
  if (row.owner !== owner || typeof row.id !== 'string' || !row.id || typeof row.suite !== 'string' || typeof row.version !== 'string') return;
  const report = browserReport(row.report,row.id), accepted = browserReport(row.acceptedReport,row.id);
  return {owner,id:row.id,suite:row.suite,version:row.version,updated:typeof row.updated === 'string' ? row.updated : '',report,
    acceptedReport:accepted && !accepted.error && !accepted.mutation && accepted.tests.length && accepted.tests.every(test=>test.passed) ? accepted : undefined};
}

let database: Promise<IDBDatabase> | undefined;
function open() {
  return database ||= new Promise<IDBDatabase>((resolve,reject) => {
    const request = indexedDB.open('paperleet.learning.v1',2);
    let blocked=false;
    request.onblocked=()=>{blocked=true;reject(Error('Close an older tab to update the learning cache'));};
    request.onupgradeneeded = () => {
      const db=request.result,create=()=>db.createObjectStore('results',{keyPath:['owner','id','suite']});
      if(!db.objectStoreNames.contains('results')){create();return;}
      const before=request.transaction!.objectStore('results').getAll();
      before.onsuccess=()=>{
        try{
          db.deleteObjectStore('results');const store=create();
          for(const value of before.result){
            if(typeof value?.owner!=='string')continue;
            const row=parseBrowserLearning(value,value.owner);if(row)store.put(row);
          }
        }catch{request.transaction?.abort();}
      };
    };
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {const db=request.result;if(blocked){db.close();return;}db.onversionchange=()=>{db.close();database=undefined;};resolve(db);};
  }).catch(error=>{database=undefined;throw error;});
}
const value = <T>(request: IDBRequest<T>) => new Promise<T>((resolve,reject)=>{request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
async function transaction<T>(owner: string, action: (store: IDBObjectStore)=>Promise<T>) {
  if (!learningOwner(owner)) throw Error('Unresolved browser learning owner');
  const db=await open(), tx=db.transaction('results','readwrite');
  const done=new Promise<void>((resolve,reject)=>{tx.oncomplete=()=>resolve();tx.onabort=()=>reject(tx.error || Error('Learning save aborted'));tx.onerror=()=>reject(tx.error);});
  void done.catch(()=>{});
  try {const result=await action(tx.objectStore('results'));await done;return result;}
  catch(error){try{tx.abort();}catch{/* Already completed or aborted. */}await done.catch(()=>{});throw error;}
}
export function loadBrowserLearning(owner: string) {
  return transaction(owner,async store=>Object.fromEntries((await value<unknown[]>(store.getAll()))
    .map(row=>parseBrowserLearning(row,owner)).filter((row):row is BrowserLearning=>!!row).map(row=>[learningKey(row.id,row.suite),row])));
}
export function saveBrowserLearning(owner: string, problem: Problem, report: Report, version: string) {
  return transaction(owner,async store=>{
    const parsed=browserReport(report,problem.id);
    if(!parsed)throw Error('Invalid browser learning report');
    const previous=parseBrowserLearning(await value(store.get([owner,problem.id,problem.browser_judge?.bundle_url || ''])),owner);
    if(previous?.version===version)return previous;
    const record=mergeBrowserLearning(previous,owner,problem,parsed,version);store.put(record);return record;
  });
}
/** A native result supersedes only the browser report this tab has observed. */
export function dismissBrowserLearning(owner: string, id: string, suite:string, expected: string, version: string) {
  return transaction(owner,async store=>{
    const previous=parseBrowserLearning(await value(store.get([owner,id,suite])),owner);
    if(previous?.version!==expected)return previous;
    const record={...previous,report:undefined,version};store.put(record);return record;
  });
}
