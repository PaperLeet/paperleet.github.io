import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
try {
  const { publicationQuotaMessage } = await server.ssrLoadModule('/src/publicationQuota.ts');
  const { PublishingRules, DAILY_PUBLICATION_LIMITS } = await server.ssrLoadModule('/src/PublishingRules.tsx');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const original = globalThis.localStorage;
  try {
    for (const english of [false, true]) {
      globalThis.localStorage = { getItem: () => english ? 'en' : 'zh' };
      for (const [action, limit] of Object.entries(DAILY_PUBLICATION_LIMITS)) {
        const response = { code: 'daily_quota_exceeded', quota: { action, limit, reset_at: '2026-10-15T00:00:00+00:00' } };
        const message = publicationQuotaMessage(response, english);
        assert(message.includes(String(limit)) && message.includes(english ? 'Your draft is kept' : '草稿已保留'));
        const html = renderToString(React.createElement(LocaleProvider, null, React.createElement(PublishingRules, { action })));
        assert(html.includes('UTC') && html.includes(english ? 'Deletion does not restore quota' : '删除内容不返还额度'));
        assert(html.includes(english ? 'human review' : '人工评审'));
        assert(html.includes(english ? 'Code submissions have separate limits' : '做题代码提交单独计算'));
      }
    }
  } finally {
    if (original === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = original;
  }
  for (const data of [{}, { code: 'other' }, { code: 'daily_quota_exceeded', quota: { action: 'admin', limit: 10, reset_at: '2026-01-01' } },
    { code: 'daily_quota_exceeded', quota: { action: 'paper', limit: 1, reset_at: 'invalid' } }]) {
    assert.equal(publicationQuotaMessage(data, true), undefined, 'Invalid error data must use the normal API error fallback');
  }
  console.log('PASS publication rules: bilingual limits and reset explanations, separate code submissions, review policy, and safe quota error handling');
} finally { await server.close(); }
