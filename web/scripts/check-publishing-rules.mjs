import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom', optimizeDeps: {noDiscovery: true, include: []} });
try {
  const { publicationQuotaMessage } = await server.ssrLoadModule('/src/publicationQuota.ts');
  const { socialApi, SocialError } = await server.ssrLoadModule('/src/social.tsx');
  const { PublishingRules, DAILY_PUBLICATION_LIMITS } = await server.ssrLoadModule('/src/PublishingRules.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { canPublishCommunity } = await server.ssrLoadModule('/src/communityPublishing.ts');
  const work = {id:'worker',status:'active',email_verified:true,institution_verified:false,institution:null,affiliation:{kind:'email_domain',domain:'new-research-company.example',name:'New Research Company'}};
  assert(canPublishCommunity(work), 'Verified work-domain affiliation needs no institutional catalog entry');
  assert(!canPublishCommunity({...work,can_publish_community:false}), 'Explicit server denial overrides other positive profile flags');
  assert(canPublishCommunity({id:'server-verified',can_publish_community:true}), 'Authoritative server eligibility supports incremental response rollout');
  assert(!canPublishCommunity({...work,status:'waitlisted',can_publish_community:true}));
  assert(!canPublishCommunity({...work,affiliation:null}), 'A verified personal email with no affiliation cannot publish');
  assert(!canPublishCommunity({...work,email_verified:false}), 'Display affiliation alone is insufficient');
  const legacyInstitution={id:'known',email_verified:true,institution_verified:true,institution:{id:'example',name:'Example University'}};
  assert(canPublishCommunity(legacyInstitution));
  assert(!canPublishCommunity({...legacyInstitution,email_verified:false}), 'Contradictory legacy institution metadata cannot replace verified email');
  assert(!canPublishCommunity({...legacyInstitution,email_verified:undefined}));
  assert(!canPublishCommunity({...legacyInstitution,institution_verified:false}));
  assert(!canPublishCommunity(null));
  assert(canPublishCommunity({id:'personal',status:'active',affiliation:null},{community_open_publishing:true}), 'Current services allow active accounts without an affiliation badge');
  assert(!canPublishCommunity({id:'personal',can_publish_community:false},{community_open_publishing:true}), 'An explicit account denial remains authoritative');
  assert(canPublishCommunity({id:'dev',is_test_user:true},{auth_mode:'test'}));
  assert(!canPublishCommunity({id:'dev',is_test_user:true,can_publish_community:true},{auth_mode:'oauth'}), 'Test markers only bypass verification in explicit test mode');
  assert(!canPublishCommunity({id:'dev',is_test_user:true,can_publish_community:false},{auth_mode:'test'}));
  const original = globalThis.localStorage, originalFetch = globalThis.fetch;
  try {
    for (const english of [false, true]) {
      globalThis.localStorage = { getItem: () => english ? 'en' : 'zh' };
      for (const [action, limit] of Object.entries(DAILY_PUBLICATION_LIMITS)) {
        const response = { code: 'daily_quota_exceeded', quota: { action, limit, reset_at: '2026-10-15T00:00:00+00:00' } };
        const message = publicationQuotaMessage(response, english);
        assert(message.includes(String(limit)) && message.includes(english ? 'Your draft is kept' : '草稿已保留'));
        globalThis.fetch = async () => Response.json({...response,error:'Server quota rejection'},{status:429});
        await assert.rejects(socialApi('community/posts',{title:'Retained draft'}),error=>error instanceof SocialError && error.status===429 && error.code===response.code && error.message===message,'Deferred wording preserves the typed API rejection and current locale');
        const html = renderToString(React.createElement(LocaleProvider, null, React.createElement(PublishingRules, { action })));
        assert(html.includes('UTC') && html.includes(english ? 'Deletion does not restore quota' : '删除内容不返还额度'));
        assert(html.includes(english ? 'human review' : '人工评审'));
        assert(html.includes(english ? 'Code submissions have separate limits' : '做题代码提交单独计算'));
      }
      const profileLimit={code:'profile_edit_limit',field:'display_name',next_change_at:'2030-10-21T00:00:00Z',error:'Profile limit'};
      globalThis.fetch=async()=>Response.json(profileLimit,{status:429});
      await assert.rejects(socialApi('account/profile',{display_name:'Retained name'}),error=>error instanceof SocialError && error.profileEditField==='display_name' && error.nextChangeAt===profileLimit.next_change_at && error.message.includes(english?'Your edits are kept':'当前编辑内容仍保留'),'Deferred profile errors keep recovery fields and draft guidance');
    }
    for(const code of ['daily_quota_exceeded','profile_edit_limit','account_changed']){
      globalThis.fetch=async()=>Response.json({code,error:'Original server rejection'},{status:409});
      await assert.rejects(socialApi('account/profile',{}),error=>error instanceof SocialError && error.status===409 && error.code===code && error.message==='Original server rejection','Unrecognized or incomplete detail uses the server message');
    }
    globalThis.fetch=async()=>Response.json({user_id:'current-owner'});
    assert.deepEqual(await socialApi('account/me'),{user_id:'current-owner'});
  } finally {
    globalThis.fetch=originalFetch;
    if (original === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = original;
  }
  for (const data of [{}, { code: 'other' }, { code: 'daily_quota_exceeded', quota: { action: 'admin', limit: 10, reset_at: '2026-01-01' } },
    { code: 'daily_quota_exceeded', quota: { action: 'paper', limit: 1, reset_at: 'invalid' } }]) {
    assert.equal(publicationQuotaMessage(data, true), undefined, 'Invalid error data must use the normal API error fallback');
  }
  console.log('PASS publication rules: authoritative affiliation eligibility, verified unknown work domains, legacy proof contradictions, test-mode-only exception, bilingual limits, reset explanations and safe quota handling');
} finally { await server.close(); }
