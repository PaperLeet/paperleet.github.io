import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {build} from 'esbuild';
import {JSDOM} from 'jsdom';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import ReactMarkdown from 'react-markdown';

const escape = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link = (href,label) => `<a href="${escape(href)}">${escape(label)}</a>`;
const sourceLink = paper => /^https?:\/\//.test(paper.url) ? link(paper.url,paper.arxiv?`Read paper · arXiv:${paper.arxiv}`:'Read source material') : '';
const stylesheet = `#prerender-catalog{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;color:inherit}#prerender-catalog header{display:flex;align-items:center;justify-content:space-between;gap:20px;border-bottom:1px solid #bbb;padding-bottom:16px}#prerender-catalog header img{width:40px;height:40px;vertical-align:middle;margin-right:10px}#prerender-catalog h1{font-size:30px;line-height:1.25;margin:28px 0 12px}#prerender-catalog h2{font-size:22px;margin-top:28px}#prerender-catalog a{color:#a05f00}#prerender-catalog li{margin:12px 0}#prerender-catalog pre{overflow:auto;padding:16px;background:#8881}#prerender-catalog .catalog-grid{list-style:none;padding:0;display:grid;gap:12px}#prerender-catalog .catalog-grid li{border:1px solid #8884;border-radius:12px;padding:16px;margin:0}#prerender-catalog .catalog-grid p{margin:6px 0}#prerender-catalog .source{color:#777}#prerender-catalog .startup-status{font-size:13px;border-top:1px solid #8884;margin-top:32px;padding-top:12px}#prerender-catalog .startup-status p{margin:4px 0}#prerender-catalog .startup-status [hidden]{display:none}`;

export async function loadSearchTools() {
  const result=await build({entryPoints:['src/searchMetadata.ts'],bundle:true,platform:'node',format:'esm',write:false});
  const metadata=await import('data:text/javascript;base64,'+Buffer.from(result.outputFiles[0].text).toString('base64'));
  const collections=await build({entryPoints:['src/paperCollections.ts'],bundle:true,platform:'node',format:'esm',write:false});
  const reading=JSON.parse(await fs.readFile('src/data/collection-reading-lists.json','utf8'));
  return {...metadata,...await import('data:text/javascript;base64,'+Buffer.from(collections.outputFiles[0].text).toString('base64')),collectionReadingLists:reading.collections};
}

function bodyFor(meta,catalog,tools) {
  const {paperPath,problemPath,paperCollections}=tools;
  const paperCards=rows=>`<ul class="catalog-grid">${rows.map(paper=>`<li>${link(paperPath(paper),`${paper.number}. ${paper.short_title}`)}<p>${escape(paper.summary_en)}</p><small>${escape(paper.venue||paper.year||'')} · ${paper.module_ids.length} exercises</small></li>`).join('')}</ul>`;
  let body='';
  if (meta.kind==='home') {
    body=`<p>LeetCode for AI Researchers. Implement algorithms from research papers in Python and test your understanding.</p><nav aria-label="Featured collections">${paperCollections.map(row=>link(`/collections/${row.id}/`,row.name_en)).join(' · ')}</nav>${paperCards([...catalog.papers].sort((a,b)=>a.number-b.number).slice(0,20))}`;
  } else if (meta.kind==='collection') {
    const collection=paperCollections.find(row=>row.id===meta.id);
    const papers=collection.paperIds.flatMap(id=>catalog.papers.filter(paper=>paper.id===id));
    body=`<p>${escape(meta.description)}</p>`+(collection.kind==='course'
      ? `<ul class="catalog-grid">${papers.flatMap(paper=>paper.module_ids).map(id=>catalog.problems.find(row=>row.id===id)).filter(Boolean).map(problem=>`<li>${link(problemPath(problem),problem.translations?.en?.title||problem.title)}<p>${escape(problem.module.goal_en)}</p></li>`).join('')}</ul>`
      : paperCards(papers));
    const reading=tools.collectionReadingLists?.find(row=>row.id===collection.id);
    if(reading) {
      const count=reading.stages.reduce((total,stage)=>total+stage.papers.length,0);
      body=`<p>${escape(meta.description)}</p><p>${escape(reading.description_en)}</p><h2>Reading list · ${count} papers</h2>`+
        reading.stages.map(stage=>`<section><h3>${escape(stage.title_en)}</h3><ul class="catalog-grid">${stage.papers.map(entry=>{
          const practice=entry.paper_id&&papers.find(paper=>paper.id===entry.paper_id);
          const source=/^https:\/\//.test(entry.url)?link(entry.url,entry.title):escape(entry.title);
          return `<li data-reading-paper="${escape(entry.id)}"><strong>${escape(entry.short_title)}</strong><p>${source}</p><small>${escape(entry.year)}${entry.priority?' · Recommended next':''} · ${practice?'Practice available':'No exercise yet'}</small><p>${escape(entry.summary_en)}</p>${practice?link(paperPath(practice),'Practice '+entry.short_title):''}</li>`;
        }).join('')}</ul></section>`).join('');
    }
  } else if (meta.kind==='paper') {
    const paper=catalog.papers.find(row=>row.id===meta.id);
    const modules=paper.module_ids.map(id=>catalog.problems.find(row=>row.id===id));
    if (modules.some(row=>!row)) throw Error(`Missing module for ${paper.id}`);
    body=`<p>${escape(paper.title)}</p><p class="source">${escape(Array.isArray(paper.authors)?paper.authors.join(', '):paper.authors)}${paper.venue?` · ${escape(paper.venue)}`:''}</p><p>${sourceLink(paper)}</p><p>${escape(paper.summary_en)}</p><h2>What you will implement</h2><p>${escape(paper.scope_en)}</p><ul class="catalog-grid">${modules.map(problem=>`<li>${link(problemPath(problem),`${problem.number}. ${problem.translations?.en?.title||problem.title}`)}<p>${escape(problem.module.goal_en)}</p><p>${escape(problem.module.scope_en)}</p></li>`).join('')}</ul>`;
  } else {
    const problem=catalog.problems.find(row=>row.id===meta.id),paper=catalog.papers.find(row=>row.id===problem.paper_id);
    body=`<p>${link(paperPath(paper),paper.short_title)} · ${escape(paper.venue||'')} · ${escape({E:'Easy',M:'Medium',H:'Hard'}[problem.difficulty]||problem.difficulty)}${problem.estimated_minutes?` · ${problem.estimated_minutes} minutes`:''}</p><p>${escape(problem.module.goal_en)}</p><p>${escape(problem.module.scope_en)}</p><h2>Functions to implement</h2><ul>${problem.module.required_functions.map(name=>`<li><code>${escape(problem.functions.find(fn=>fn.name===name)?.signature||name)}</code></li>`).join('')}</ul><section aria-label="Problem description">${renderToStaticMarkup(React.createElement(ReactMarkdown,{skipHtml:true},problem.translations?.en?.readme||problem.readme))}</section><p>${sourceLink(paper)}</p><h2>Related exercises</h2><ul>${paper.module_ids.filter(id=>id!==problem.id).map(id=>catalog.problems.find(row=>row.id===id)).filter(Boolean).map(row=>`<li>${link(problemPath(row),row.translations?.en?.title||row.title)}</li>`).join('')}<li>${link(paperPath(paper),`All ${paper.short_title} exercises`)}</li></ul>`;
  }
  return `<main id="prerender-catalog"><header>${link('/', 'PaperLeet')}<nav aria-label="Main navigation">${link('/','Papers')} · ${link('/#/community','Community')} · ${link('/#/interviews','Interviews')}</nav></header><h1>${escape(meta.heading)}</h1>${body}<aside class="startup-status" aria-live="polite"><strong id="startup-title">Opening PaperLeet…</strong><p id="startup-message" role="status">Loading interactive practice.</p><a id="startup-retry" href="?reload=1" hidden>Reload page</a><noscript><p>You can read the paper and exercise here. Enable JavaScript to run Python and save your progress.</p></noscript></aside></main>`;
}

export function renderPage(template,meta,catalog,tools) {
  const dom=new JSDOM(template),d=dom.window.document;
  d.title=meta.title;
  for(const [key,name,content] of [
    ['name','description',meta.description],['property','og:title',meta.title],['property','og:description',meta.description],
    ['property','og:url',meta.canonical],['name','twitter:title',meta.title],['name','twitter:description',meta.description],
  ]) {
    const el=d.head.querySelector(`meta[${key}="${name}"]`);if(!el)throw Error(`Missing ${name}`);el.content=content;
  }
  d.head.querySelector('link[rel="canonical"]').href=meta.canonical;
  d.getElementById('root').innerHTML=bodyFor(meta,catalog,tools);
  const style=d.querySelector('#prerender-style')||d.createElement('style');style.id='prerender-style';style.textContent=stylesheet;d.head.append(style);
  const output=dom.serialize();dom.window.close();return output;
}

export async function prerender({directory='dist',catalogFile='public/data/catalog.json'}={}) {
  const template=await fs.readFile(path.join(directory,'index.html'),'utf8');
  if(!template.includes('data-routing="path"')) {console.log('Static catalog pages: disabled for hash/local editions');return;}
  const catalog=JSON.parse(await fs.readFile(catalogFile,'utf8')),tools=await loadSearchTools();
  for(const paper of catalog.papers)if(!/^[a-z0-9][a-z0-9_-]*$/.test(paper.id))throw Error('Invalid paper slug');
  for(const problem of catalog.problems)if(!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(problem.id))throw Error('Invalid problem ID');
  const routes=['',...catalog.papers.map(row=>'papers/'+row.id),...catalog.problems.map(row=>row.id),...tools.paperCollections.map(row=>'collections/'+row.id)];
  const urls=new Set();
  for(const route of routes) {
    const meta=tools.searchMetadata(route,catalog);if(!meta)throw Error(`Missing metadata: ${route}`);
    if(urls.has(meta.canonical))throw Error(`Duplicate static URL: ${meta.canonical}`);urls.add(meta.canonical);
    const target=path.join(directory,new URL(meta.canonical).pathname,'index.html');
    await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,renderPage(template,meta,catalog,tools));
  }
  // Generate from the current catalog, never a handwritten count or hash URL.
  await fs.writeFile(path.join(directory,'sitemap.xml'),`<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${[...urls].map(url=>`  <url><loc>${escape(url)}</loc></url>`).join('\n')}\n</urlset>\n`);
  await fs.writeFile(path.join(directory,'robots.txt'),`User-agent: *\nAllow: /\n\nSitemap: ${tools.siteOrigin}/sitemap.xml\n`);
  console.log(`Prerendered ${catalog.papers.length} paper/course pages, ${catalog.problems.length} exercises, ${tools.paperCollections.length} collections and the homepage (${urls.size} sitemap URLs)`);
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)await prerender();
