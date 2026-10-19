import assert from 'node:assert/strict';
import {act} from 'react';
export async function checkExerciseCompanySources({fixture,catalog,nav,route,wait,click,summary}){
 const request=fixture.request,saved=JSON.stringify(fixture.practice.saved),target=catalog.problems.find(p=>p.id==='rl/dpo'),reads=[];
 fixture.request=async(url,body)=>{assert.equal(body,undefined);assert(url.startsWith('community/interviews/summary?'));const params=new URLSearchParams(url.split('?')[1]);reads.push(params);return summary(params.get('company')||'google',params.get('days')||'365',[target.id]);};
 const rows=()=>[...document.querySelectorAll('[data-module-id]')],launcher=()=>document.querySelector(`[data-company-problem="${target.id}"] .library-exercise-company-trigger`);
 try{
  const origin=nav.libraryHref(undefined,{...nav.emptyLibraryFilters(),view:'modules',query:String(target.number),sort:'duration',page_size:50});await route(origin);await wait(()=>rows().length===1&&launcher());assert(!reads.some(p=>p.has('problem_id')));
  assert(!rows()[0].contains(launcher()));await click(launcher());await wait(()=>document.querySelector('.library-exercise-company-filter'));assert.equal(reads.filter(p=>p.has('problem_id')).length,1);assert.equal(reads.find(p=>p.has('problem_id')).get('problem_id'),target.id);
  const link=document.querySelector('.library-exercise-company-filter'),href=link.getAttribute('href'),filters=nav.readLibraryRoute(href).filters;assert.equal(filters.company,'google');assert.equal(filters.query,String(target.number));assert.equal(filters.page_size,50);assert.equal(filters.view,'modules');assert.equal(filters.sort,undefined);assert(link.getAttribute('aria-label').includes('4 public reports from 3 contributors'));
  await act(async()=>{link.focus();link.click();});await wait(()=>nav.readLibraryRoute(location.hash)?.filters.company==='google'&&rows().length===1);assert.equal(location.hash,href);assert(!launcher());assert.equal(rows()[0].dataset.moduleId,target.id);
  await act(async()=>history.back());await wait(()=>location.hash===origin&&launcher());assert.equal(launcher().getAttribute('aria-expanded'),'false');assert.equal(reads.filter(p=>p.has('problem_id')).length,1,'Returning to the list does not eagerly read exercise sources');
  assert.equal(JSON.stringify(fixture.practice.saved),saved);assert.deepEqual(fixture.calls.runs,[]);assert.deepEqual(fixture.calls.submits,[]);assert.equal(fixture.calls.flushes,0);
  console.log('PASS actual App exercise company sources: independent lazy row action, exact problem evidence, retained query/view/page size and explicit company sort reset, native company-link history Back, closed return state and unchanged private source with zero judge/publication');
 }finally{await route('#');fixture.request=request;}
}
