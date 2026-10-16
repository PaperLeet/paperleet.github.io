const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const resolve = require('./pages-release-gate.cjs');
const sha='a'.repeat(40), next='b'.repeat(40), annotation='c'.repeat(40), nested='d'.repeat(40);
const repository='PaperLeet/paperleet.github.io';
function fixture() {
  return {
    context:{repo:{owner:'PaperLeet',repo:'paperleet.github.io'},eventName:'push',ref:'refs/tags/v0.33.0',sha,
      payload:{ref:'refs/tags/v0.33.0',created:true,deleted:false,after:sha}},
    reference:{object:{type:'commit',sha}}, tags:{}, branch:{commit:{sha}}, comparison:{status:'identical',behind_by:0}, calls:[],
  };
}
function adapter(f) {
  return {context:f.context,github:{rest:{git:{
    getRef:async args=>{f.calls.push('ref');assert.equal(args.ref,'tags/v0.33.0');return {data:f.reference};},
    getTag:async args=>{f.calls.push('tag');if(!f.tags[args.tag_sha])throw Error('Missing annotated tag');return {data:f.tags[args.tag_sha]};},
  },repos:{getBranch:async args=>{f.calls.push('main');assert.equal(args.branch,'main');return {data:f.branch};}}},
  request:async (route,args)=>{f.calls.push('compare');assert.equal(route,'GET /repos/{owner}/{repo}/compare/{basehead}');assert.equal(args.basehead,`${sha}...${f.branch.commit.sha}`);return {data:f.comparison};}}};
}
async function denied(edit,pattern) {const f=fixture();edit(f);await assert.rejects(()=>resolve(adapter(f)),pattern);}
(async()=>{
  assert.deepEqual(await resolve(adapter(fixture())),{sha,tag:'v0.33.0',tagObject:sha});
  const annotated=fixture();annotated.reference.object={type:'tag',sha:annotation};annotated.context.payload.after=annotation;annotated.tags[annotation]={object:{type:'commit',sha}};
  assert.deepEqual(await resolve(adapter(annotated)),{sha,tag:'v0.33.0',tagObject:annotation});
  annotated.tags[annotation]={object:{type:'tag',sha:nested}};annotated.tags[nested]={object:{type:'commit',sha}};
  assert.equal((await resolve(adapter(annotated))).sha,sha,'Nested annotated tags resolve to the exact tested commit');
  for(const edit of [
    f=>{f.context.eventName='workflow_run';},f=>{f.context.eventName='workflow_dispatch';},
    f=>{f.context.ref=f.context.payload.ref='refs/heads/main';},
    f=>{f.context.ref=f.context.payload.ref='refs/tags/vlatest';},
    f=>{f.context.payload.deleted=true;},f=>{f.context.payload.created=false;},
    f=>{f.context.payload.ref='refs/tags/v0.32.0';},
  ])await denied(edit,/newly created version tag/);
  await denied(f=>{f.context.repo.owner='fork';},/official repository/);
  await denied(f=>{f.context.sha=next;},/moved or.*inconsistent/);
  await denied(f=>{f.context.payload.after=next;},/moved or.*inconsistent/);
  await denied(f=>{f.reference.object={type:'tree',sha};},/resolve to a commit/);
  await denied(f=>{f.reference.object.sha='bad';},/resolve to a commit/);
  await denied(f=>{f.reference.object={type:'tag',sha:annotation};f.tags[annotation]={object:{type:'tag',sha:annotation}};},/annotated tag chain/);
  for(const status of ['behind','diverged'])await denied(f=>{f.comparison.status=status;},/contained in main/);
  await denied(f=>{f.comparison.behind_by=1;},/contained in main/);
  await denied(f=>{f.branch.commit.sha=next;},/contained in main/);
  const expected={sha,tag:'v0.33.0',tagObject:sha};
  assert.equal(await resolve.stillCurrent({...adapter(fixture()),expected}),true);
  const advanced=fixture();advanced.branch.commit.sha=next;advanced.comparison={status:'ahead',behind_by:0};
  assert.equal(await resolve.stillCurrent({...adapter(advanced),expected}),true,'Main may advance while the explicitly tagged tested source remains fixed');
  const moved=fixture();moved.reference.object.sha=next;
  assert.equal(await resolve.stillCurrent({...adapter(moved),expected}),false);
  assert.equal(await resolve.stillCurrent({...adapter(annotated),expected}),false,'An annotation rewrite cannot reuse old publication outputs');
  const ci=fs.readFileSync(path.join(__dirname,'../../.github/workflows/ci.yml'),'utf8');
  const workflow=fs.readFileSync(path.join(__dirname,'../../.github/workflows/pages.yml'),'utf8');
  const ciEvents=ci.split('on:\n')[1].split('permissions:\n')[0];
  assert.match(ciEvents,/workflow_call:/);assert.match(ciEvents,/workflow_dispatch:/);
  assert(!/^  (push|pull_request|workflow_run):/m.test(ciEvents),'Development/PR updates do not start CI');
  const events=workflow.split('on:\n')[1].split('permissions:\n')[0];
  assert.match(events,/push:\n\s+tags: \['v\*'\]/);
  assert(!/branches:|workflow_run:|workflow_dispatch:|pull_request:/.test(events),'Only a user-created version tag can enter publication');
  const release=workflow.split('  release:\n')[1].split('  quality:\n')[0];
  const condition=release.split('    if: >-\n')[1].split('    runs-on:')[0];
  const eligible=new Function('github','startsWith',`return (${condition});`);
  const event={repository,event_name:'push',ref:'refs/tags/v0.33.0',event:{created:true,deleted:false}};
  assert.equal(eligible(event,(a,b)=>a.startsWith(b)),true);
  for(const update of [e=>{e.ref='refs/heads/main';},e=>{e.event_name='pull_request';},e=>{e.event.deleted=true;},e=>{e.event.created=false;},e=>{e.repository='fork/repo';}]) {
    const e=structuredClone(event);update(e);assert.equal(eligible(e,(a,b)=>a.startsWith(b)),false);
  }
  const quality=workflow.split('  quality:\n')[1].split('  build:\n')[0];
  assert.match(quality,/needs: release/);assert.match(quality,/uses: \.\/\.github\/workflows\/ci\.yml/);
  assert(quality.includes('revision: ${{ needs.release.outputs.sha }}'));
  assert.match(workflow,/needs: \[release, quality\]/);assert.match(workflow,/needs: \[release, quality, build\]/);
  assert.equal((ci.match(/ref: \$\{\{ inputs\.revision \|\| github\.sha \}\}/g)||[]).length,3,'Every quality checkout uses the tag-resolved source');
  const native=ci.split('  native:\n')[1].split('  infrastructure:\n')[0];
  assert.match(native,/run: paperleet ci --budget 600\n/);
  assert.match(ci,/run: python -m pytest tests packages\/stridetorch\/tests -q\n/);
  const wasm=ci.split('  browser-wasm:\n')[1].split('  pilot:\n')[0];assert.match(wasm,/npm run check:browser-wasm\n/);assert.match(wasm,/PAPERLEET_WASM_CONCURRENCY: '2'/);
  const pilot=ci.split('  pilot:\n')[1];assert.match(pilot,/needs: \[native, infrastructure, browser-wasm\]/);
  for(const [key,expression] of [['NATIVE_RESULT','needs.native.result'],['INFRASTRUCTURE_RESULT','needs.infrastructure.result'],['WASM_RESULT',"needs['browser-wasm'].result"]]) {
    assert(pilot.includes(key+': $'+'{{ '+expression+' }}'),'Pilot inputs come from their actual required jobs');
  }
  const shell=pilot.split('        run: |\n')[1].replace(/^ {10}/gm,'');
  const success={NATIVE_RESULT:'success',INFRASTRUCTURE_RESULT:'success',WASM_RESULT:'success'};
  const status=values=>spawnSync('sh',['-eu','-c',shell],{env:{...process.env,...values},encoding:'utf8'}).status;
  assert.equal(status(success),0);
  for(const key of Object.keys(success))for(const value of ['failure','skipped','cancelled','timed_out',''])assert.notEqual(status({...success,[key]:value}),0);
  const deploy=workflow.split('  deploy:\n')[1];
  assert.match(deploy,/if \(!publish\) throw new Error/);assert.match(deploy,/stillCurrent/);
  for(const name of ['SHA','TAG','TAG_OBJECT'])assert(deploy.includes('RELEASE_'+name+': $'+'{{'), 'Deployment receives pinned tag outputs');
  const script=deploy.split('script: |\n')[1].split('\n      - uses: actions/deploy-pages')[0].replace(/^ {12}/gm,'');
  const execute=new (Object.getPrototypeOf(async function(){}).constructor)('require','github','context','core','process',script);
  async function beforeDeploy(f) {
    const outputs={};const a=adapter(f);
    await execute(()=>resolve,a.github,a.context,{setOutput:(k,v)=>{outputs[k]=v;},info:()=>{}},{env:{RELEASE_SHA:sha,RELEASE_TAG:'v0.33.0',RELEASE_TAG_OBJECT:sha}});
    return outputs;
  }
  assert.equal((await beforeDeploy(fixture())).publish,'true');
  await assert.rejects(()=>beforeDeploy(moved),/release tag changed/);
  assert.equal((await beforeDeploy(advanced)).publish,'true');
  assert(deploy.includes("if: steps.current.outputs.publish == 'true'"));
  assert(workflow.includes("job.steps?.some(step => step.name.includes('deploy-pages@') && step.conclusion === 'success')"),'Historical assets come from actual deployments');
  const history=workflow.split('name: Find the previous successful Pages deployment')[1].split('script: |\n')[1].split('\n      - name: Download previous static assets')[0].replace(/^ {12}/gm,'');
  const historyOutputs={};const listArtifacts=()=>{},listJobs=()=>{};
  const historyGithub={rest:{actions:{listWorkflowRunArtifacts:listArtifacts,listJobsForWorkflowRun:listJobs,listWorkflowRuns:async args=>{
    assert.equal(args.branch,undefined,'Tag deployments are included in history');return {data:{workflow_runs:[{id:1},{id:2,head_branch:'v0.32.9'}]}};
  }}},paginate:async (method,args)=>{
    if(method===listArtifacts)return [{name:'paperleet-pages-assets',expired:false,expires_at:'2099-01-01T00:00:00Z'}];
    assert.equal(method,listJobs);return [{name:'deploy',conclusion:'success',steps:[{name:'Run actions/deploy-pages@v4',conclusion:args.run_id===1?'skipped':'success'}]}];
  }};
  const findHistory=new (Object.getPrototypeOf(async function(){}).constructor)('github','context','core',history);
  await findHistory(historyGithub,{repo:{owner:'PaperLeet',repo:'paperleet.github.io'},runId:99},{info:()=>{},warning:()=>{},setOutput:(k,v)=>{historyOutputs[k]=v;}});
  assert.equal(historyOutputs.run_id,'2','Overall success with a skipped deployment cannot be the prior published asset source');
  console.log('PASS tag-only release: no branch/PR CI;manual checks do not publish;lightweight/annotated tags,main ancestry,immutable source,reused full quality,all-job failure gates and moved-tag rejection');
})().catch(error=>{console.error(error);process.exitCode=1;});
