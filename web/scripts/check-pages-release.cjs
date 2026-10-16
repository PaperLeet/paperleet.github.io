const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const resolve = require('./pages-release-gate.cjs');
const {checkReleaseRequest} = require('./check-release-request.cjs');
const sha = 'a'.repeat(40), old = 'b'.repeat(40), tagObject = 'c'.repeat(40);
const repository = 'PaperLeet/paperleet.github.io';
function fixture() {
  return {
    context: {repo:{owner:'PaperLeet',repo:'paperleet.github.io'},eventName:'workflow_run',ref:'refs/heads/main',sha,runId:70,payload:{workflow_run:{id:40}}},
    branch:{commit:{sha}},workflow:{id:7},
    run:{id:40,workflow_id:7,head_repository:{full_name:repository},head_branch:'main',head_sha:sha,event:'workflow_dispatch',status:'completed',conclusion:'success'},
    jobs:['native','infrastructure','browser-wasm','pilot'].map(name=>({name,status:'completed',conclusion:'success'})),
    tag:{object:{type:'commit',sha}}, pages:[], history:[],
  };
}
function adapter(f) {
  const actions = {
    getWorkflow:async args=>{assert.equal(args.workflow_id,'ci.yml');return {data:f.workflow};},
    getWorkflowRun:async args=>{assert.equal(args.run_id,40);return {data:f.run};},
    listWorkflowRuns:async args=>({data:{workflow_runs:args.workflow_id==='pages.yml'?f.pages:(f.runs||[f.run])}}),
    listJobsForWorkflowRun:()=>{throw Error('Use pagination');},
  };
  return {context:f.context,github:{rest:{actions,repos:{getBranch:async()=>({data:f.branch})},git:{
    getRef:async args=>{assert.equal(args.ref,`tags/${f.run.head_branch}`);return {data:f.tag};},
    getTag:async args=>{assert.equal(args.tag_sha,tagObject);return {data:{object:{type:'commit',sha:f.tagTarget||sha}}};},
  }},paginate:async(method,args)=>{if(method===actions.listWorkflowRuns){assert.equal(args.workflow_id,'pages.yml');return f.pages;}assert.equal(method,actions.listJobsForWorkflowRun);assert.equal(args.filter,'latest');return args.run_id===40?f.jobs:f.history;}}};
}
async function denied(edit,pattern) {const f=fixture();edit(f);await assert.rejects(()=>resolve(adapter(f)),pattern);}
(async()=>{
  // Execute release admission against fresh remote refs, including annotated tags.
  const env={GITHUB_REPOSITORY:repository,GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_REF:'refs/heads/main',GITHUB_SHA:sha};
  function git(...args) {
    if(args[0]==='rev-parse')return sha;
    if(args.includes('refs/heads/main'))return `${sha}\trefs/heads/main`;
    return `${tagObject}\trefs/tags/v1.2.3\n${sha}\trefs/tags/v1.2.3^{}`;
  }
  assert.equal(checkReleaseRequest({env,git}),sha);
  const tagged={...env,GITHUB_EVENT_NAME:'push',GITHUB_REF:'refs/tags/v1.2.3',GITHUB_SHA:tagObject};
  assert.equal(checkReleaseRequest({env:tagged,git}),sha);
  for(const change of [{GITHUB_EVENT_NAME:'pull_request'},{GITHUB_EVENT_NAME:'push'},{GITHUB_REF:'refs/heads/feature'},
    {GITHUB_REPOSITORY:'someone/fork'},{GITHUB_SHA:'main'},{GITHUB_SHA:sha+'\n'}])assert.throws(()=>checkReleaseRequest({env:{...env,...change},git}));
  for(const ref of ['v1','v01.2.3','v1.2.3-rc1','main','v1.2.3\n'])assert.throws(()=>checkReleaseRequest({env:{...tagged,GITHUB_REF:'refs/tags/'+ref},git}));
  assert.throws(()=>checkReleaseRequest({env,git:(...a)=>a[0]==='ls-remote'?`${old}\trefs/heads/main`:sha}),/current main/);
  assert.throws(()=>checkReleaseRequest({env:tagged,git:(...a)=>a.includes(tagged.GITHUB_REF)?`${old}\t${tagged.GITHUB_REF}`:git(...a)}),/moved or deleted/);
  assert.throws(()=>checkReleaseRequest({env,git:(...a)=>a[0]==='rev-parse'&&a[1]!=='HEAD'?old:git(...a)}),/Checkout/);

  assert.deepEqual(await resolve(adapter(fixture())),{sha,runId:40});
  const tag=fixture();tag.run.event='push';tag.run.head_branch='v1.2.3';
  assert.deepEqual(await resolve(adapter(tag)),{sha,runId:40});
  tag.tag={object:{type:'tag',sha:tagObject}};
  assert.deepEqual(await resolve(adapter(tag)),{sha,runId:40},'Annotated tags peel to the tested commit');
  tag.tagTarget=old;await assert.rejects(()=>resolve(adapter(tag)),/moved or deleted/);
  const manual=fixture();manual.context.eventName='workflow_dispatch';
  manual.runs=[{...manual.run,id:99,head_sha:old},{...manual.run,id:98,event:'pull_request'},{...manual.run,id:97,event:'push'},manual.run];
  assert.deepEqual(await resolve(adapter(manual)),{sha,runId:40},'Ignore old ordinary-push and PR quality runs');
  for(const [key,value] of [['conclusion','failure'],['conclusion','cancelled'],['conclusion','timed_out'],['status','in_progress'],['head_sha',old],['head_branch','feature'],['event','pull_request'],['workflow_id',8]])await denied(f=>{f.run[key]=value;},/successful complete explicit quality/);
  await denied(f=>{f.run.event='push';},/successful complete explicit quality/);
  await denied(f=>{f.run.head_repository.full_name='someone/fork';},/successful complete explicit quality/);
  await denied(f=>{delete f.run.head_repository;},/successful complete explicit quality/);
  await denied(f=>{f.branch.commit.sha=old;},/successful complete explicit quality/);
  await denied(f=>{f.context.repo.owner='fork';},/official repository/);
  await denied(f=>{f.context.eventName='push';},/explicit release/);
  await denied(f=>{f.context.eventName='workflow_dispatch';f.context.ref='refs/heads/feature';},/explicit release/);
  await denied(f=>{f.context.eventName='workflow_dispatch';f.runs=[];},/complete explicit release/);
  await denied(f=>{f.context.payload.workflow_run.id='40';},/complete explicit release/);
  for(const name of ['native','infrastructure','browser-wasm','pilot']) {
    await denied(f=>{f.jobs=f.jobs.filter(j=>j.name!==name);},/missing or unsuccessful/);
    for(const conclusion of ['skipped','failure','cancelled',null])await denied(f=>{f.jobs.find(j=>j.name===name).conclusion=conclusion;},/missing or unsuccessful/);
    await denied(f=>{f.jobs.push({...f.jobs.find(j=>j.name===name)});},/missing or unsuccessful/);
  }
  const duplicate=fixture();duplicate.pages=[{id:50,head_sha:sha}];
  duplicate.history=[{name:'deploy',conclusion:'success',steps:[{name:'Run actions/deploy-pages@v4',conclusion:'success'}]}];
  assert.deepEqual(await resolve(adapter(duplicate)),{sha,runId:40,publish:false});
  duplicate.history[0].steps[0].conclusion='skipped';
  assert.deepEqual(await resolve(adapter(duplicate)),{sha,runId:40},'Overall success with skipped deploy is not publication');

  const workflow=fs.readFileSync(path.join(__dirname,'../../.github/workflows/pages.yml'),'utf8');
  assert(!/^  push:/m.test(workflow)&&!/^  pull_request:/m.test(workflow));
  assert(!/branches: \[main\]/.test(workflow),'Tag completion events must not be filtered out');
  assert.match(workflow,/publish: \$\{\{ steps\.release\.outputs\.publish \}\}/);
  assert.match(workflow,/if: needs\.build\.outputs\.publish == 'true'/);
  assert.match(workflow,/if: steps\.current\.outputs\.publish == 'true'/);
  const condition=workflow.split('    if: >-\n')[1].split('\n    runs-on:')[0];
  const eligible=new Function('github','startsWith',`return (${condition});`);
  const event=()=>({repository,event_name:'workflow_run',ref:'refs/heads/main',sha,event:{workflow_run:{...fixture().run}}});
  const allowed=e=>eligible(e,(value,prefix)=>typeof value==='string'&&value.startsWith(prefix));
  assert.equal(allowed(event()),true);
  const tagEvent=event();tagEvent.event.workflow_run.event='push';tagEvent.event.workflow_run.head_branch='v1.2.3';assert.equal(allowed(tagEvent),true);
  for(const edit of [e=>e.event.workflow_run.event='pull_request',e=>e.event.workflow_run.event='push',e=>e.event.workflow_run.head_sha=old,e=>e.repository='someone/fork',e=>e.event_name='push']){const e=event();edit(e);assert.equal(allowed(e),false);}
  const script=workflow.split('name: Recheck the tested revision before publishing')[1].split('script: |\n')[1].split('\n      - uses:')[0].replace(/^ {12}/gm,'');
  const execute=new (Object.getPrototypeOf(async function(){}).constructor)('github','context','core','process','require',script);
  async function beforeDeploy(edit){const f=fixture();edit(f);const outputs={};await execute(adapter(f).github,f.context,{setOutput:(k,v)=>outputs[k]=v,info(){}},{env:{RELEASE_SHA:sha,QUALITY_RUN:'40'}},()=>resolve);return outputs.publish;}
  assert.equal(await beforeDeploy(()=>{}),'true');
  await assert.rejects(()=>beforeDeploy(f=>{f.branch.commit.sha=old;}),/successful complete/);
  await assert.rejects(()=>beforeDeploy(f=>{f.run.conclusion='failure';}),/successful complete/);
  assert.equal(await beforeDeploy(f=>{f.pages=duplicate.pages;f.history=[{name:'deploy',conclusion:'success',steps:[{name:'Run actions/deploy-pages@v4',conclusion:'success'}]}];}),'false');
  const ci=fs.readFileSync(path.join(__dirname,'../../.github/workflows/ci.yml'),'utf8');
  assert(!/^  pull_request:/m.test(ci)&&!/^    branches:/m.test(ci));assert.match(ci,/tags: \['v\*'\]/);
  assert.equal((ci.match(/run: node web\/scripts\/check-release-request.cjs/g)||[]).length,3);
  assert.match(ci,/run: paperleet ci --budget 600/);assert.match(ci,/run: python -m pytest tests packages\/stridetorch\/tests -q/);assert.match(ci,/npm run check:browser-wasm/);
  const pilot=ci.split('  pilot:\n')[1];assert.match(pilot,/needs: \[native, infrastructure, browser-wasm\]/);
  const shell=pilot.split('        run: |\n')[1].replace(/^ {10}/gm,'');
  const success={NATIVE_RESULT:'success',INFRASTRUCTURE_RESULT:'success',WASM_RESULT:'success'};
  const status=values=>spawnSync('sh',['-eu','-c',shell],{env:{...process.env,...values}}).status;
  assert.equal(status(success),0);
  for(const key of Object.keys(success))for(const value of ['failure','cancelled','skipped','timed_out',''])assert.notEqual(status({...success,[key]:value}),0);
  console.log('PASS explicit release gates: manual/tag-only, exact current main, annotated/moved tags, complete quality, provenance, duplicate publication and stale/rerun rejection');
})().catch(error=>{console.error(error);process.exitCode=1;});
