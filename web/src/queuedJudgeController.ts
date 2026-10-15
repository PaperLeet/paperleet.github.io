/** Deferred server-queue lifecycle, isolated to one active workspace owner. */
import { JobRequestError, requestJob, watchJob } from './judgeJobs';
import type { JudgeJob } from './judgeJobs';
import type { Report } from './types';
import type { AdmissionWait } from './queueAdmission';

type Pending = { id:string; problem:string };
export type QueuedJudgeView = { owner:string; pending?:Pending; job?:JudgeJob; waiting?:AdmissionWait; reconnecting:boolean; error:string };
export const pendingKey = (owner:string) => `paperleet.judge.pending.v1:${owner}`;
const empty = (owner:string):QueuedJudgeView => ({owner,reconnecting:false,error:''});
type Options = { owner:string; client:string; expectedUser?:string; storage?:Pick<Storage,'getItem'|'setItem'|'removeItem'>;
  alive:()=>boolean; onView:(view:QueuedJudgeView)=>void; onReport:(report:Report,owner:string)=>void;
  fetcher?:typeof fetch; wait?:typeof import('./judgeJobs').pollDelay; pollWait?:typeof import('./judgeJobs').pollDelay;
  now?:()=>number; wall?:()=>number; uuid?:()=>string };

export class QueuedJudgeController {
  private view:QueuedJudgeView;
  private controller?:AbortController;
  private busy=false;
  private waiting=false;
  private disposed=false;
  constructor(private options:Options) { this.view=empty(options.owner); }
  private owns(signal:AbortSignal) { return !this.disposed&&!signal.aborted&&this.options.alive()&&this.controller?.signal===signal; }
  private emit(view:QueuedJudgeView) { if(!this.disposed&&this.options.alive()){this.view=view;this.options.onView(view);} }
  private remember(pending?:Pending) {
    try {
      const storage=this.options.storage||sessionStorage;
      if(pending)storage.setItem(pendingKey(this.options.owner),JSON.stringify(pending));
      else storage.removeItem(pendingKey(this.options.owner));
    } catch { /* A stored receipt can only reconnect with GET; it never resubmits code. */ }
  }
  private recall():Pending|undefined {
    try {
      const value=JSON.parse((this.options.storage||sessionStorage).getItem(pendingKey(this.options.owner))||'null');
      if(value&&/^[a-f0-9]{32}$/.test(value.id)&&typeof value.problem==='string')return {id:value.id,problem:value.problem};
    } catch { /* Malformed/blocked storage cannot authorize a submission. */ }
  }
  private clear(signal:AbortSignal) { if(this.controller?.signal===signal){this.busy=false;this.waiting=false;} }
  private async attach(pending:Pending,signal:AbortSignal) {
    const {owner,client}=this.options;
    try {
      const job=await watchJob(pending.id,client,signal,job=>{
        if(job.problem!==pending.problem)throw new JobRequestError('The judge returned a different module',422);
        if(this.owns(signal))this.emit({...empty(owner),pending,job});
      },connected=>{if(this.owns(signal))this.emit({...this.view,reconnecting:!connected});},this.options.fetcher,this.options.pollWait);
      if(!this.owns(signal))return;
      this.remember();
      this.emit({...empty(owner),job,error:job.status==='failed'?job.error||'Judge job failed':''});
      if(job.status==='succeeded'&&job.report)this.options.onReport(job.report,owner);
    } catch(error) {
      if(!this.owns(signal))return;
      this.remember();
      this.emit({...empty(owner),error:error instanceof Error?error.message:'Judge job unavailable'});
    } finally {this.clear(signal);}
  }
  resume() {
    if(this.busy||this.disposed||!this.options.alive())return;
    const pending=this.recall();if(!pending)return;
    this.controller=new AbortController();this.busy=true;
    this.emit({...empty(this.options.owner),pending});
    void this.attach(pending,this.controller.signal);
  }
  async submit(problem:string,code:string) {
    if(this.busy||this.disposed||!this.options.alive())return;
    if(this.recall()){this.resume();return;}
    const {owner,client,expectedUser}=this.options;
    const pending={id:this.options.uuid?.()||crypto.randomUUID().replaceAll('-',''),problem};
    const next=new AbortController();this.controller?.abort();this.controller=next;this.busy=true;
    this.emit({...empty(owner),pending});
    let tracking=false;
    try {
      const admission=await import('./queueAdmission').catch(()=>{
        throw new JobRequestError('Server submission could not start. Your draft is preserved; try again.',400);
      });
      if(!this.owns(next.signal))return;
      const job=await admission.admitSubmission({...pending,code},client,next.signal,expectedUser,()=>{
        if(!this.owns(next.signal))throw new DOMException('Owner changed','AbortError');
        this.waiting=false;this.remember(pending);
        this.emit({...empty(owner),pending,...(this.view.waiting?{waiting:{...this.view.waiting,attempting:true}}:{})});
      },state=>{
        if(!this.owns(next.signal))throw new DOMException('Owner changed','AbortError');
        // An explicit refusal proves this ID was not admitted. Refresh can stop
        // waiting without storing source or restarting automatic POSTs.
        this.remember();this.waiting=true;this.emit({...empty(owner),pending,waiting:state});
      },this.options);
      if(!this.owns(next.signal))return;
      if(job.id!==pending.id||job.problem!==pending.problem)throw new JobRequestError('The judge returned a different submission',502);
      this.emit({...empty(owner),pending,job});
      tracking=true;void this.attach(pending,next.signal);
    } catch(error) {
      if(!this.owns(next.signal))return;
      this.waiting=false;
      if(!(error instanceof JobRequestError)||error.status>=500||error.status===422) {
        this.remember(pending);this.emit({...empty(owner),pending,reconnecting:true});
        tracking=true;void this.attach(pending,next.signal);
      } else {this.remember();this.emit({...empty(owner),error:error.message});}
    } finally {if(!tracking)this.clear(next.signal);}
  }
  stopWaiting() {
    const signal=this.controller?.signal;
    if(!signal||!this.owns(signal))return false;
    if(!this.waiting){this.emit({...this.view,error:'The submission is being confirmed. Wait for its queue status before cancelling.'});return false;}
    this.controller!.abort();this.clear(signal);this.remember();this.emit(empty(this.options.owner));return true;
  }
  async cancel() {
    const pending=this.view.pending,signal=this.controller?.signal;
    if(!pending||!signal||!this.owns(signal)||this.view.job?.status!=='queued')return;
    try {await requestJob(`/${pending.id}/cancel`,this.options.client,signal,
      this.options.expectedUser?{expected_user_id:this.options.expectedUser}:{},this.options.fetcher);
    } catch(error) {if(this.owns(signal))this.emit({...this.view,error:error instanceof Error?error.message:'Could not cancel this job'});}
  }
  dispose(){this.disposed=true;this.controller?.abort();this.busy=false;this.waiting=false;}
}
