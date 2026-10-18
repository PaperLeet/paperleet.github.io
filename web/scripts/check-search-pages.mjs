import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {JSDOM} from 'jsdom';
import {loadSearchTools,renderPage} from './prerender.mjs';

const home=await fs.readFile('dist/index.html','utf8');
if(!home.includes('data-routing="path"')) {console.log('Static search checks: not a path-routing edition');process.exit(0);}
const catalog=JSON.parse(await fs.readFile('public/data/catalog.json','utf8')),tools=await loadSearchTools();
const sitemap=await fs.readFile('dist/sitemap.xml','utf8');
const xml=new JSDOM(sitemap,{contentType:'application/xml'});
const urls=[...xml.window.document.querySelectorAll('loc')].map(el=>el.textContent);xml.window.close();
assert.equal(urls.length,1+catalog.papers.length+catalog.problems.length+tools.paperCollections.length);
assert.equal(new Set(urls).size,urls.length);
assert((await fs.readFile('dist/robots.txt','utf8')).includes('Sitemap: https://paperleet.com/sitemap.xml'));

// Model a directory-based static host: missing documents must remain real 404s.
const directory=path.resolve('dist');
const server=http.createServer(async(req,res)=>{
  try {
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname),target=path.resolve(directory,'.'+pathname);
    if(!target.startsWith(directory+'/')&&target!==directory)throw Error('Outside build');
    const stat=await fs.stat(target);
    if(stat.isDirectory()&&!pathname.endsWith('/')) {res.writeHead(301,{Location:pathname+'/'});res.end();return;}
    const body=await fs.readFile(stat.isDirectory()?path.join(target,'index.html'):target);
    res.writeHead(200);res.end(body);
  } catch {res.writeHead(404);res.end('Not found');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`,assets=new Set(),titles=new Set();
try {
  for(const url of urls) {
    const parsed=new URL(url);assert.equal(parsed.origin,tools.siteOrigin);assert(!parsed.hash&&!parsed.search);
    const response=await fetch(origin+parsed.pathname);assert.equal(response.status,200,url);
    const html=await response.text(),dom=new JSDOM(html),d=dom.window.document;
    assert.equal(d.querySelector('link[rel="canonical"]').href,url);
    assert(!titles.has(d.title),`Duplicate title: ${d.title}`);titles.add(d.title);
    assert(d.title.includes('PaperLeet'));
    assert(d.querySelector('meta[name="description"]').content.length>40);
    assert.equal(d.querySelector('meta[property="og:title"]').content,d.title);
    assert.equal(d.querySelector('meta[property="og:url"]').content,url);
    assert.equal(d.querySelector('meta[property="og:image"]').content,tools.siteOrigin+'/brand/social-card.png');
    assert.equal(d.querySelector('meta[name="twitter:card"]').content,'summary_large_image');
    assert(d.querySelector('#prerender-catalog h1')?.textContent.trim());
    assert(d.querySelector('#prerender-catalog').textContent.length>180,'Static page must contain usable content without JS');
    assert.equal(d.querySelectorAll('link[rel="canonical"]').length,1);
    if(parsed.pathname.startsWith('/problems/')) {
      const id=parsed.pathname.slice(10,-1),problem=catalog.problems.find(row=>row.id===id);assert(problem);
      assert(d.body.textContent.includes(problem.module.goal_en));
      for(const name of problem.module.required_functions)assert(d.body.textContent.includes(name));
    }
    for(const element of d.querySelectorAll('script[src],link[rel="stylesheet"],link[rel="icon"]')) {
      const asset=element.getAttribute('src')||element.getAttribute('href');
      assert(asset.startsWith('/'),`Deep page asset is relative: ${asset}`);assets.add(asset);
    }
    for(const anchor of d.querySelectorAll('#prerender-catalog a[href]')) {
      const href=anchor.getAttribute('href');
      if(/^\/(papers|problems|collections)\//.test(href))assert(urls.includes(tools.siteOrigin+href),`Broken static catalog link: ${href}`);
    }
    dom.window.close();
  }
  for(const asset of assets)assert.equal((await fetch(origin+asset)).status,200,asset);
  const png=await fs.readFile('dist/brand/social-card.png');assert.equal(png.readUInt32BE(16),1200);assert.equal(png.readUInt32BE(20),630);
  assert.equal((await fetch(origin+'/papers/flow-matching',{redirect:'manual'})).status,301);
  assert.equal((await fetch(origin+'/papers/flow-matching')).status,200);
  assert.equal((await fetch(origin+'/papers/not-a-real-paper/')).status,404);
  assert.equal((await fetch(origin+'/problems/not-a-real-module/')).status,404);
  const poisoned=structuredClone(catalog),paper=poisoned.papers[0];
  paper.short_title='Example </title><script id="injected">bad()</script>';
  paper.summary_en='<img src=x onerror="bad()">';
  const html=renderPage(home,tools.searchMetadata('papers/'+paper.id,poisoned),poisoned,tools),safe=new JSDOM(html);
  assert.equal(safe.window.document.querySelector('#injected'),null);
  assert.equal(safe.window.document.querySelector('[onerror]'),null);
  assert(safe.window.document.body.textContent.includes(paper.summary_en));safe.window.close();
  console.log(`PASS ${urls.length} static HTTP 200 routes, distinct metadata, no-JS content, root assets, sitemap, sharing image, escaping and real missing-page 404s`);
} finally {server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
