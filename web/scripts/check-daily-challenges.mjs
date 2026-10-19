import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createServer} from 'vite';
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try {
  const {challengeMonth,dailySelection,dailyRoute,dailySchedule,readDailySnapshot}=await server.ssrLoadModule('/src/dailyChallengeModel.ts');
  const {companyPracticeMatches}=await server.ssrLoadModule('/src/ExternalCompanyPractice.tsx');
  const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
  const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
  const catalog=JSON.parse(fs.readFileSync('public/data/index.json'));
  for(const id of dailySchedule.modules)assert(catalog.problems.some(row=>row.id===id),'Unavailable scheduled module '+id);
  assert.equal(challengeMonth('2028-02').length,29);assert.equal(challengeMonth('2027-02').length,28);assert.equal(challengeMonth('2026-13').length,0);assert.equal(challengeMonth('invalid').length,0);
  assert.equal(dailySelection('2026-10-19').problem_id,'architecture/attention');
  assert.equal(dailySelection('2026-10-20').problem_id,'foundations/next_token_loss');
  assert.equal(dailySelection('2026-10-18'),undefined);assert.equal(dailySelection('2026-02-30'),undefined);
  assert.equal(dailySelection('2026-11-16').problem_id,'architecture/attention','Date mapping remains stable across cycles');
  assert.deepEqual(dailyRoute('daily-challenge?date=2026-10-19'),{date:'2026-10-19'});
  assert.equal(dailyRoute('daily-challenge?date=2026-10-19&rating=99999'),undefined);
  for(const href of ['#/daily-challenge','#/daily-challenge?date=2026-10-19'])assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))),href);
  const data={challenge:dailySelection('2026-10-19'),server_time:'2026-10-19T12:00:00Z',personal:{user_id:'alice',rating:1000,completed_days:0,entry:null,history:[]}};
  assert(readDailySnapshot(data,'alice'));assert.equal(readDailySnapshot(data,'bob'),undefined);assert.equal(readDailySnapshot({...data,personal:{...data.personal,rating:NaN}},'alice'),undefined);
  const external=JSON.parse(fs.readFileSync('src/data/external-company-practice.json'));
  assert.equal(external.items.length,50);assert.equal(new Set(external.items.map(row=>row.id)).size,50);assert.equal(new Set(external.items.map(row=>row.company)).size,43);
  for(const row of external.items){assert(new URL(row.url).origin==='https://trentorch.com');assert(/^\/ide\/[a-z0-9-]+-company-\d+$/.test(new URL(row.url).pathname));for(const id of row.related_modules)assert(catalog.problems.some(row=>row.id===id));}
  assert(companyPracticeMatches('AWS','amazon'));assert(companyPracticeMatches('Google','google'));assert(companyPracticeMatches('Netflix','other'));assert(!companyPracticeMatches('Meta','other'));assert(!companyPracticeMatches('OpenAI','google'));
  console.log('Daily calendar, schedule integrity, account continuations, owner isolation and 50 attributed company exercises passed.');
}finally{await server.close();}
