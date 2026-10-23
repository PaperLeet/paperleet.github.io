import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM,VirtualConsole} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm,readFile,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
// CSS helpers: comments dropped and whitespace normalized; outsideMedia drops every @media and @container block (rules that hold
// at every width), media returns the body of one @media or @container block, and decls lists the declarations of one exact selector.
const normalizeCss=text=>text.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\s*([{};:,>])\s*/g,'$1').replace(/;}/g,'}');
const blockEnd=(css,open)=>{let depth=0,end=open;for(;end<css.length;end++)if(css[end]==='{')depth++;else if(css[end]==='}'&&!--depth)break;return end;};
const media=(css,prelude)=>{const start=css.indexOf(prelude+'{');return start<0?'':css.slice(start+prelude.length+1,blockEnd(css,start+prelude.length));};
const outsideMedia=css=>{for(let at;(at=css.search(/@(media|container)/))>=0;)css=css.slice(0,at)+css.slice(blockEnd(css,css.indexOf('{',at))+1);return css;};
const decls=(css,selector)=>[...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter(match=>match[1].trim()===selector).flatMap(match=>match[2].split(';'));
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-collection-roadmap-'),server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try{
 const nav=await server.ssrLoadModule('/src/libraryNavigation.ts');
 // Section: route state (view=roadmap)
 const empty=nav.emptyLibraryFilters(),roadmap={...empty,view:'roadmap'},href='#/collections/jepa-world-models?view=roadmap';
 assert.deepEqual(nav.readLibraryRoute(href),{collectionId:'jepa-world-models',filters:{query:'',category:'all',tag:'all',difficulty:'all',runtime:'all',state:'all',view:'roadmap'}});
 assert.equal(nav.libraryHref('jepa-world-models',roadmap),href);
 assert.equal(nav.libraryReturnHref(nav.withLibraryContext('papers/i-jepa',href)),href);
 assert.equal('#/'+nav.libraryContinuation(href),href);
 for(const bad of ['#/?view=roadmap','#/collections/llm-basics?view=roadmap','#/collections/rl?view=roadmap&q=x','#/collections/rl?view=roadmap&page=2','#/collections/rl?view=roadmap&sort=newest','#/collections/rl?view=roadmap&category=reinforcement-learning','#/collections/rl?view=roadmap&view=roadmap','#/collections/rl?view=ROADMAP'])assert.equal(nav.readLibraryRoute(bad),undefined,bad);
 assert.deepEqual(nav.normalizeLibraryFilters({...roadmap,difficulty:'E',page:2,company:'openai'},[]),roadmap);
 assert.equal(nav.readLibraryRoute('#/collections/rl?view=modules').filters.view,'modules');
 console.log('PASS collection roadmap routes: exclusive view=roadmap, rejected combinations, return context');

 // Section: roadmap model helpers
 const data=await server.ssrLoadModule('/src/collectionRoadmaps.ts'),model=await server.ssrLoadModule('/src/roadmapModel.ts');
 assert.equal(data.roadmapFor('jepa-world-models')?.collectionId,'jepa-world-models');
 assert.equal(data.roadmapFor('nope'),undefined);
 const node=(id,year,row,col)=>({id,title:id,year,venue:'V',sentence:'s',sentence_en:'s',row,col}),[A,B,C,D]=[node('A',2020,0,0),node('B',2021,1,0),node('C',2021,1,1),node('D',2023,2,0)];
 const synth={collectionId:'x',nodes:[A,B,C,D],edges:[['A','B'],['A','C'],['B','D']]};
 assert.deepEqual(model.nodeBox({row:0,col:0}),{x:70,y:52,width:126,height:50});
 assert.equal(model.nodeBox({row:0,col:1.5}).x,58+1.5*150+12);
 // Start tag: above the node when nothing points at it, else to its left.
 const sb=model.nodeBox(A),above=model.startTag({collectionId:'x',nodes:[{...A,start:true},B],edges:[['A','B']]});
 assert.deepEqual(model.START_TAG,{width:96,height:22,gap:6,side:8});
 assert.equal(above.side,'above');assert.deepEqual([above.left,above.top],[sb.x+sb.width/2,sb.y]);
 assert.deepEqual(above.box,{x:sb.x+sb.width/2-48,y:sb.y-6-22,width:96,height:22});
 const lb=model.nodeBox(B),left=model.startTag({collectionId:'x',nodes:[A,{...B,start:true}],edges:[['A','B']]});
 assert.equal(left.side,'left');assert.deepEqual([left.left,left.top],[lb.x-8,lb.y+lb.height/2]);
 assert.equal(left.box.x+left.box.width,lb.x-8);assert.equal(left.box.y+left.box.height/2,lb.y+lb.height/2);assert.deepEqual([left.box.width,left.box.height],[96,22]);
 assert.deepEqual(model.roadmapSize(synth),{width:58+2*150+16,height:34+3*86+16});
 assert.ok(model.edgePath(A,B).startsWith('M 133,102 C'),model.edgePath(A,B));
 const pts=model.edgePoints(A,B),bb=model.nodeBox(B);
 assert.equal(pts.length,25);
 assert.deepEqual(pts[0],{x:133,y:102});
 assert.deepEqual(pts[24],{x:bb.x+bb.width/2,y:bb.y});
 assert.deepEqual(model.yearRows(synth),[{year:2020,row:0},{year:2021,row:1},{year:2023,row:2}]);
 assert.deepEqual([...model.neighbors(synth,'B')].sort(),['A','B','D']);
 const papers=JSON.parse(await readFile('public/data/catalog.json','utf8')).papers;
 assert.equal(model.matchPaper({paper:'i-jepa'},papers)?.id,'i-jepa');
 assert.equal(model.matchPaper({arxiv:'2106.09685'},papers)?.id,'lora');
 assert.equal(model.matchPaper({arxiv:'2404.08471'},papers),undefined);
 assert.equal(model.matchPaper({paper:'i-jepa'},papers.map(p=>p.id==='i-jepa'?{...p,kind:'course'}:p)),undefined);
 console.log('PASS collection roadmap model: geometry, edges, year rows, neighbors, paper matching');

 // Section: curated data rules (membership, caps, chronology, sparse arrows, no clutter)
 const {paperCollections}=await server.ssrLoadModule('/src/paperCollections.ts'),catalogIds=new Set(papers.map(p=>p.id));
 assert.deepEqual(data.collectionRoadmaps.map(r=>r.collectionId).sort(),paperCollections.filter(c=>c.roadmap).map(c=>c.id).sort(),'Exactly one roadmap per collection marked roadmap: true, and none for the rest');
 assert(paperCollections.every(c=>!(c.roadmap&&c.kind==='course')),'A course has no roadmap');
 for(const c of paperCollections)assert.equal(!!nav.readLibraryRoute(`#/collections/${c.id}?view=roadmap`),c.kind!=='course',`${c.id}: every featured collection has a roadmap view, curated or not, and a course has none`);
 // Clutter rules on the page's own geometry: each arrow is the 24-segment polyline of edgePoints, each node its nodeBox grown by
 // CLEAR px (its zone). Segments intersect with a 1e-6 orientation tolerance, so touching counts as crossing.
 // - Pass-through: no arrow segment may enter the zone of a node other than its own two ends.
 // - Crossing: any two arrows are compared. Arrows that share one node (fan-out, fan-in or a chain) leave or enter that node at the
 //   same point, so segments with an end inside the shared node's zone are dropped; outside it, any contact counts. Over every
 //   fan-out and fan-in pair on a 12-row grid this agrees with a sign test of the exact curves (20,001 samples each) for any zone from 2 to
 //   13 px, except two pairs whose curves stay within 0.2 px (under the 1.5 px stroke), which count as touching. Arrows that share
 //   both nodes would be duplicates, which the edge rules reject.
 const CLEAR=4,orient=(p,q,r)=>{const v=(q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x);return Math.abs(v)<1e-6?0:Math.sign(v);},cross=(a,b,c,d)=>orient(a,b,c)!==orient(a,b,d)&&orient(c,d,a)!==orient(c,d,b);
 const zone=n=>{const b=model.nodeBox(n);return {x0:b.x-CLEAR,y0:b.y-CLEAR,x1:b.x+b.width+CLEAR,y1:b.y+b.height+CLEAR};},inside=(v,z)=>v.x>z.x0&&v.x<z.x1&&v.y>z.y0&&v.y<z.y1;
 const hitsZone=([p,q],z)=>{const c=[{x:z.x0,y:z.y0},{x:z.x1,y:z.y0},{x:z.x1,y:z.y1},{x:z.x0,y:z.y1}];return inside(p,z)||inside(q,z)||c.some((v,i)=>cross(p,q,v,c[(i+1)%4]));};
 const layoutProblems=r=>{
  const byId=new Map(r.nodes.map(n=>[n.id,n])),out=[];
  const arrows=r.edges.map(([from,to])=>{const pts=model.edgePoints(byId.get(from),byId.get(to));return {from,to,segs:pts.slice(1).map((q,k)=>[pts[k],q])};});
  for(let i=0;i<arrows.length;i++)for(let j=i+1;j<arrows.length;j++){
   const e=arrows[i],f=arrows[j],shared=[e.from,e.to].find(id=>id===f.from||id===f.to),z=shared&&zone(byId.get(shared));
   const away=segs=>z?segs.filter(s=>!s.some(v=>inside(v,z))):segs,fs=away(f.segs);
   if(away(e.segs).some(([p,q])=>fs.some(([u,v])=>cross(p,q,u,v))))out.push(`arrows ${e.from}->${e.to} and ${f.from}->${f.to} cross`);
  }
  for(const e of arrows)for(const n of r.nodes){
   if(n.id===e.from||n.id===e.to)continue;
   const z=zone(n);if(e.segs.some(s=>hitsZone(s,z)))out.push(`arrow ${e.from}->${e.to} passes through ${n.id}`);
  }
  return out;
 };
 {
  const cell=(id,row,col)=>({id,row,col}),layout=(nodes,edges)=>layoutProblems({collectionId:'synthetic',nodes,edges});
  // (a) Fan-out S-curves that cross away from their shared source: the shipped diffusion arrows Score SDE -> Flow Matching / Consistency.
  assert.deepEqual(layout([cell('score-sde',3,2),cell('flow-matching',6,1),cell('consistency',7,0)],[['score-sde','flow-matching'],['score-sde','consistency']]),['arrows score-sde->flow-matching and score-sde->consistency cross'],'Fan-out arrows that cross below their shared node are flagged');
  // (a') The same pair rotated 180 degrees: fan-in arrows that cross above their shared target.
  assert.deepEqual(layout([cell('c',3,3),cell('f',4,2),cell('s',7,1)],[['f','s'],['c','s']]),['arrows f->s and c->s cross'],'Fan-in arrows that cross above their shared node are flagged');
  // (b) Two arrows with no node in common that swap columns.
  assert.deepEqual(layout([cell('a',0,0),cell('b',0,1),cell('c',1,0),cell('d',1,1)],[['a','d'],['b','c']]),['arrows a->d and b->c cross'],'Disjoint arrows that cross are flagged');
  // (c) An arrow drawn straight through the node between its ends.
  assert.deepEqual(layout([cell('a',0,0),cell('b',1,0),cell('c',2,0)],[['a','c']]),['arrow a->c passes through b'],'An arrow through another node is flagged');
  // Clean: a diamond has a fan-out, a fan-in and chained arrows, and none of them cross or touch away from their shared node.
  assert.deepEqual(layout([cell('a',0,1),cell('b',1,0),cell('c',1,2),cell('d',2,1),cell('e',3,1)],[['a','b'],['a','c'],['b','d'],['c','d'],['d','e']]),[],'A clean fork-and-join layout has no problems');
  console.log('PASS collection roadmap clutter rules: fan-out, fan-in and disjoint crossings and pass-through flagged; clean fork-and-join accepted');
 }
 for(const r of data.collectionRoadmaps){
  const at=`${r.collectionId}: `,byId=new Map(r.nodes.map(n=>[n.id,n]));
  assert.ok(r.nodes.length>=3&&r.nodes.length<=14,at+'3 to 14 papers');
  assert.equal(byId.size,r.nodes.length,at+'unique node ids');
  const rows=new Set();
  for(const n of r.nodes){
   assert.match(n.id,/^[a-z0-9]+(?:-[a-z0-9]+)*$/,at+n.id);
   assert.ok(!n.paper||catalogIds.has(n.paper),at+n.id+' paper exists in the catalog');
   assert.ok(typeof n.arxiv==='string'&&/^\d{4}\.\d{4,5}$/.test(n.arxiv),at+n.id+' has an arXiv id without version (maps hold arXiv papers only)');
   assert.ok(Number.isInteger(n.row)&&n.row>=0&&Number.isInteger(n.col*2)&&n.col>=0&&n.col<=3,at+n.id+' grid cell (col in half steps within 0-3)');
   rows.add(n.row);
   assert.ok(n.title&&n.venue&&Number.isInteger(n.year),at+n.id+' title, venue and year');
   assert.ok(n.sentence.length>=1&&n.sentence.length<=70,at+n.id+' zh sentence length');
   assert.ok(n.sentence_en.length>=1&&n.sentence_en.length<=140&&!/\n/.test(n.sentence_en),at+n.id+' en sentence length');
  }
  assert.deepEqual([...rows].sort((a,b)=>a-b),[...Array(rows.size).keys()],at+'contiguous rows from 0');
  for(const a of r.nodes)for(const b of r.nodes){
   if(a.row===b.row)assert.equal(a.year,b.year,at+`${a.id}/${b.id} share a row but not a year`);
   if(a!==b&&a.row===b.row)assert.ok(Math.abs(a.col-b.col)>=1,at+`${a.id}/${b.id} sit less than 1 column apart`);
   if(a.year<b.year)assert.ok(a.row<b.row,at+`${a.id} is earlier than ${b.id} but not above it`);
  }
  const keys=new Set(),incoming=new Map();
  for(const [from,to,zh,en] of r.edges){
   assert.equal(r.edges.find(e=>e[0]===from&&e[1]===to).length,4,at+`edge ${from}->${to} has 4 entries`);
   assert.ok(typeof zh==='string'&&zh.length>=1&&zh.length<=40&&zh.endsWith('。'),at+`edge ${from}->${to} zh sentence`);
   assert.ok(typeof en==='string'&&en.length>=1&&en.length<=90&&en.endsWith('.')&&!/\n/.test(en),at+`edge ${from}->${to} en sentence`);
   assert.ok(byId.has(from)&&byId.has(to),at+`edge ${from}->${to} references existing nodes`);
   assert.ok(!keys.has(from+'>'+to),at+'duplicate edge');keys.add(from+'>'+to);
   assert.ok(byId.get(from).row<byId.get(to).row,at+`edge ${from}->${to} points downward`);
   incoming.set(to,(incoming.get(to)||0)+1);
  }
  for(const [id,count] of incoming)assert.ok(count<=2,at+id+' has at most 2 incoming arrows');
  const reach=(from,to,skip)=>{const seen=new Set(),stack=[from];while(stack.length){const cur=stack.pop();for(const [a,b] of r.edges){if(a!==cur||a+'>'+b===skip||seen.has(b))continue;if(b===to)return true;seen.add(b);stack.push(b);}}return false;};
  for(const [from,to] of r.edges)assert.ok(!reach(from,to,from+'>'+to),at+`edge ${from}->${to} skips a path that already exists`);
  const starts=r.nodes.filter(n=>n.start);
  assert.equal(starts.length,1,at+'exactly one start node');
  const tag=model.startTag(r),tb=tag.box,size=model.roadmapSize(r);
  assert.equal(tag.side,incoming.has(starts[0].id)?'left':'above',at+'start tag side follows incoming arrows');
  assert.ok(tb.x>=model.ROADMAP.gutter-4&&tb.y>=0&&tb.x+tb.width<=size.width&&tb.y+tb.height<=size.height,at+'start tag inside the canvas');
  for(const n of r.nodes){const b=model.nodeBox(n);assert.ok(tb.x+tb.width<=b.x||b.x+b.width<=tb.x||tb.y+tb.height<=b.y||b.y+b.height<=tb.y,at+`start tag overlaps ${n.id}`);}
  for(const [from,to] of r.edges)for(const q of model.edgePoints(byId.get(from),byId.get(to)))assert.ok(!(q.x>=tb.x&&q.x<=tb.x+tb.width&&q.y>=tb.y&&q.y<=tb.y+tb.height),at+`start tag covers arrow ${from}->${to}`);
  assert.deepEqual(layoutProblems(r),[],at+'no crossing or pass-through arrows');
 }
 console.log('PASS collection roadmap data: one roadmap per collection marked roadmap: true, a roadmap view on every featured collection and none on a course, 3-14 papers, an arXiv id on every node, half-step columns, catalog ids, chronology, sparse downward arrows without skips, one start with a clear tag box, short bilingual node and arrow sentences, no crossings or pass-through');

 // Section: paper cards. Every node has card metadata and a small local WebP cover (RIFF size field = file size - 8) 168px wide,
 // twice the 84px card cover, and neither the metadata nor the thumbnail folder holds anything a map does not use. The width comes
 // from the image header: lossy VP8 (14 bits at 26), lossless VP8L (14 bits plus 1 at 21) or extended VP8X (24 bits plus 1 at 24).
 const arxivIds=[...new Set(data.collectionRoadmaps.flatMap(r=>r.nodes.map(n=>n.arxiv)))].sort();
 const webpWidth=bytes=>({'VP8 ':()=>bytes.readUInt16LE(26)&0x3fff,'VP8L':()=>(bytes.readUInt32LE(21)&0x3fff)+1,'VP8X':()=>bytes.readUIntLE(24,3)+1})[bytes.toString('latin1',12,16)]?.();
 assert.deepEqual(Object.keys(data.roadmapPapers).sort(),arxivIds,'roadmapPapers holds exactly the arXiv ids of the maps');
 for(const [id,paper] of Object.entries(data.roadmapPapers)){
  assert.ok(typeof paper.title==='string'&&paper.title.trim()===paper.title&&paper.title.length>0,id+' title');
  assert.ok(Array.isArray(paper.authors)&&paper.authors.length>=1&&paper.authors.length<=6&&paper.authors.every(author=>typeof author==='string'&&author.trim()),id+' 1 to 6 authors');
  assert.ok(/^\d{4}-\d{2}-\d{2}$/.test(paper.published)&&new Date(paper.published+'T00:00:00Z').toISOString().startsWith(paper.published),id+' YYYY-MM-DD date');
 }
 const thumbs=(await readdir('public/roadmap-thumbnails').catch(()=>[])).sort();
 assert.deepEqual(thumbs,arxivIds.map(id=>id+'.webp'),'One thumbnail per arXiv id of the maps and nothing else');
 for(const file of thumbs){const bytes=await readFile('public/roadmap-thumbnails/'+file);assert.ok(bytes.length<=16384,file+' is at most 16384 bytes');assert.ok(bytes.toString('latin1',0,4)==='RIFF'&&bytes.toString('latin1',8,12)==='WEBP'&&bytes.readUInt32LE(4)===bytes.length-8,file+' is a WebP file');assert.equal(webpWidth(bytes),168,file+' is 168px wide');}
 // The paper object exactly as CollectionRoadmap builds it (date at local midnight) passes readPaperPreview whole for every id, so
 // no card falls back to "Paper details are unavailable".
 const {readPaperPreview}=await server.ssrLoadModule('/src/communityPapers.ts'),{parseArxivUrl}=await server.ssrLoadModule('/src/arxivLinks.ts');
 for(const [id,meta] of Object.entries(data.roadmapPapers)){
  const arxiv=parseArxivUrl('https://arxiv.org/abs/'+id),paper={arxiv_id:arxiv.arxivId,url:arxiv.url,pdf_url:arxiv.pdfUrl,status:'ready',title:meta.title,authors:meta.authors,published:meta.published+'T00:00'};
  assert.deepEqual(readPaperPreview(paper,arxiv.url),paper,id+' card paper passes readPaperPreview with its title and authors');
 }
 console.log(`PASS collection roadmap paper cards: metadata (title, 1-6 authors, date) that readPaperPreview accepts whole as the component builds it, and a 168px WebP thumbnail of at most 16 KB for each of the ${arxivIds.length} arXiv ids, nothing else`);
}finally{await server.close();await rm(cacheDir,{recursive:true,force:true});}

// Section: CollectionRoadmap component (live StrictMode DOM, local and public editions)
{
 const errors=[],consoleErrors=[],consoleError=console.error,virtualConsole=new VirtualConsole();virtualConsole.on('jsdomError',error=>errors.push(error));console.error=(...args)=>consoleErrors.push(args.join(' '));
 const returnTo='#/collections/jepa-world-models?view=roadmap',dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/'+returnTo,pretendToBeVisual:true,virtualConsole});
 const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,location:dom.window.location,history:dom.window.history,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,requestAnimationFrame:callback=>setTimeout(callback,0),cancelAnimationFrame:clearTimeout,IS_REACT_ACT_ENVIRONMENT:true};
 const original=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
 localStorage.setItem('paperleet.locale','en');
 // A time zone west of UTC, where a date-only string read as UTC midnight would show the day before.
 const env={VITE_PUBLIC_SITE:process.env.VITE_PUBLIC_SITE,VITE_HOSTED_APP_AVAILABLE:process.env.VITE_HOSTED_APP_AVAILABLE,TZ:process.env.TZ},servers=[];let root;process.env.TZ='America/Los_Angeles';
 const serve=async()=>{const cacheDir=await mkdtemp(tmpdir()+'/paperleet-collection-roadmap-ui-'),server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});servers.push({server,cacheDir});return server;};
 try{
  const local=await serve(),{createRoot}=await import('react-dom/client'),h=React.createElement;
  const load=async server=>({Roadmap:(await server.ssrLoadModule('/src/CollectionRoadmap.tsx')).default,...await server.ssrLoadModule('/src/i18n.tsx'),deployment:await server.ssrLoadModule('/src/deployment.ts')});
  const data=await local.ssrLoadModule('/src/collectionRoadmaps.ts'),model=await local.ssrLoadModule('/src/roadmapModel.ts'),nav=await local.ssrLoadModule('/src/libraryNavigation.ts'),{paperModules}=await local.ssrLoadModule('/src/libraryPractice.ts'),{paperCollections}=await local.ssrLoadModule('/src/paperCollections.ts');
  const {papers,problems}=JSON.parse(await readFile('public/data/catalog.json','utf8')),roadmap=data.roadmapFor('jepa-world-models'),match=item=>model.matchPaper(item,papers),byId=id=>roadmap.nodes.find(item=>item.id===id);
  // Fixed anchor: I-JEPA; the first node without an exercise and everything else follow the data.
  const ijepa=roadmap.nodes.find(item=>match(item)?.id==='i-jepa'),missing=roadmap.nodes.find(item=>!match(item));
  assert(ijepa&&missing,'The roadmap needs the I-JEPA node and a node without an exercise');
  let ui,props,setLocale;
  function Frame(){setLocale=ui.useI18n().setLocale;return h(ui.Roadmap,props);}
  const render=async next=>{props={collectionId:'jepa-world-models',papers,problems,saved:{},progressReady:true,returnTo,...next};await act(async()=>{root??=createRoot(document.getElementById('root'));root.render(h(React.StrictMode,null,h(ui.LocaleProvider,null,h(Frame))));});};
  const $=selector=>document.querySelector(selector),$$=selector=>[...document.querySelectorAll(selector)],node=id=>$(`.roadmap-node[data-node="${id}"]`),panel=()=>$('aside.roadmap-panel'),canvas=()=>$('.collection-roadmap > .roadmap-canvas-scroll > .roadmap-canvas');
  const press=async element=>{assert(element);await act(async()=>{element.focus();element.click();});};
  const pair=edge=>edge.getAttribute('data-from')+'>'+edge.getAttribute('data-to'),dimmed=selector=>$$(selector).filter(element=>element.hasAttribute('data-dimmed')).map(element=>element.getAttribute('data-node')||pair(element));
  const idleText=()=>{const loose=[],walker=document.createTreeWalker($('.collection-roadmap'),dom.window.NodeFilter.SHOW_TEXT);while(walker.nextNode())if(walker.currentNode.textContent.trim()&&!walker.currentNode.parentElement.closest('.roadmap-node-title,.roadmap-year,.roadmap-start'))loose.push(walker.currentNode.textContent);assert.deepEqual(loose,[],'Idle roadmap text is only years, titles and the start tag');};
  ui=await load(local);assert.equal(ui.deployment.publicSite,false);
  await render();
  const size=model.roadmapSize(roadmap),{top,rowHeight}=model.ROADMAP,svg=canvas()?.querySelector(':scope > svg');
  assert(canvas(),'Canvas sits in the horizontal scroller');assert.equal(canvas().getAttribute('role'),'group');assert.equal(canvas().getAttribute('aria-label'),'Roadmap');assert.deepEqual([canvas().style.width,canvas().style.height],[size.width+'px',size.height+'px']);
  assert.deepEqual($$('.roadmap-node').map(element=>element.dataset.node),roadmap.nodes.map(item=>item.id),'One node button per roadmap node');
  for(const item of roadmap.nodes){const element=node(item.id),box=model.nodeBox(item);assert.equal(element.tagName,'BUTTON');assert.equal(element.type,'button');assert.equal(element.getAttribute('aria-pressed'),'false');assert.deepEqual([element.style.left,element.style.top,element.style.width,element.style.height],[box.x,box.y,box.width,box.height].map(value=>value+'px'),item.id);assert.equal(element.querySelector('.roadmap-node-title').textContent,item.title);}
  // The start tag sits where startTag puts it (above a start without incoming arrows, left of one with them), just before its node.
  const checkStart=(r,side)=>{const tag=model.startTag(r),element=$('.roadmap-start');assert.equal($$('.roadmap-start').length,1);assert.equal(element.textContent,'Start here');assert.equal(tag.side,side,r.collectionId);assert.equal(element.dataset.side,side);assert.deepEqual([element.style.left,element.style.top],[tag.left+'px',tag.top+'px'],r.collectionId+' start tag anchor');assert.equal(element.nextElementSibling,node(r.nodes.find(item=>item.start).id),'The tag is read right before its node');};
  checkStart(roadmap,'above');
  // Three arrowheads: neutral for idle edges, one per lineage direction; idle edges all use the neutral one and nothing flows.
  const markers=()=>$$('marker').map(marker=>[marker.id,marker.getAttribute('class')]),base=$('marker')?.id;
  assert(base?.endsWith('-arrow'));assert.deepEqual(markers(),[[base,'roadmap-arrow'],[base+'-from','roadmap-arrow from'],[base+'-to','roadmap-arrow to']]);
  const idleEdges=()=>{assert($$('path.roadmap-edge').every(edge=>edge.getAttribute('marker-end')===`url(#${base})`),'Idle edges end in the neutral arrow');assert(!$('.roadmap-flow,.collection-roadmap [data-dir],.collection-roadmap [data-lineage],.collection-roadmap [data-lit],.collection-roadmap [data-dimmed]'),'Nothing is lit, dimmed or flowing without a selection');};
  idleEdges();
  const diffusion=data.roadmapFor('diffusion'),ddpm=diffusion.nodes.find(item=>item.start);
  assert.deepEqual([ddpm.id,ddpm.col],['ddpm',1.5],'Diffusion starts at DDPM, centered');
  await render({collectionId:'diffusion'});checkStart(diffusion,'left');await render();
  assert.equal(svg?.getAttribute('aria-hidden'),'true');
  assert.deepEqual($$('text.roadmap-year').map(text=>[text.textContent,text.getAttribute('x'),text.getAttribute('y')]),model.yearRows(roadmap).map(({year,row})=>[String(year),'8',String(top+row*rowHeight+rowHeight/2)]));
  assert.deepEqual($$('path.roadmap-edge').map(pair),roadmap.edges.map(([from,to])=>from+'>'+to));
  for(const edge of $$('path.roadmap-edge')){assert.equal(edge.getAttribute('d'),model.edgePath(byId(edge.getAttribute('data-from')),byId(edge.getAttribute('data-to'))));const marker=edge.getAttribute('marker-end')?.match(/^url\(#(.+)\)$/)?.[1];assert.equal(marker&&document.getElementById(marker)?.localName,'marker','Every edge ends in an arrow marker');}
  assert(!panel(),'No panel until a node is selected');assert.deepEqual([...dimmed('.roadmap-node'),...dimmed('.roadmap-edge')],[]);idleText();
  console.log('PASS collection roadmap layout: canvas size, node boxes, start tag at startTag (above on JEPA, left of DDPM on Diffusion), three arrow markers, year labels, idle edges neutral and still, no idle text');

  // Phones: a canvas wider than its scroller opens with the start node's center in the middle, clamped to the scroll range, on
  // mount and on every map change; a selection or a scroller without overflow is left alone. jsdom has no layout, so the scroller
  // gets a fake width (viewport) and the canvas width as its scroll width; jsdom keeps an assigned scrollLeft as is. A root of its
  // own mounts fresh. Every map, one after another, is shown on a 358px phone scroller, a 600px one, one 10px narrower than its
  // canvas and a 1000px one (wider than any canvas); the expected offset follows the model (the start node's box center less half
  // the scroller, clamped to the scroll range), and together the maps must center a start, hit both clamps and leave one alone.
  let viewport;const phone=document.body.appendChild(document.createElement('div')),phoneRoot=createRoot(phone),scroller=()=>phone.querySelector('.roadmap-canvas-scroll');
  const show=collectionId=>act(async()=>phoneRoot.render(h(React.StrictMode,null,h(ui.LocaleProvider,null,h(ui.Roadmap,{collectionId,papers,problems,saved:{},progressReady:true,returnTo})))));
  for(const name of ['clientWidth','scrollWidth']){const get=Object.getOwnPropertyDescriptor(dom.window.Element.prototype,name).get;Object.defineProperty(dom.window.HTMLElement.prototype,name,{configurable:true,get(){return viewport&&this.classList.contains('roadmap-canvas-scroll')?name==='clientWidth'?viewport:Math.max(viewport,parseFloat(this.firstElementChild.style.width)):get.call(this);}});}
  const outcomes=new Set(),ids=data.collectionRoadmaps.map(r=>r.collectionId),showAt=async(collectionId,width)=>{
   const r=data.roadmapFor(collectionId),box=model.nodeBox(r.nodes.find(item=>item.start)),range=model.roadmapSize(r).width-width,want=Math.min(range,Math.max(0,box.x+box.width/2-width/2));
   viewport=width;if(scroller())scroller().scrollLeft=7;await show(collectionId);
   const outcome=range<=0?'left alone':want===0?'left clamp':want===range?'right clamp':'centered';outcomes.add(outcome);
   assert.equal(scroller().scrollLeft,range<=0?7:want,`${collectionId} on a ${width}px scroller: ${outcome}`);
  };
  for(const width of [358,600,undefined,1000])for(const id of ids)await showAt(id,width??model.roadmapSize(data.roadmapFor(id)).width-10);
  assert.deepEqual([...outcomes].sort(),['centered','left alone','left clamp','right clamp'],'The maps cover a centered start, both clamps and a scroller without overflow');
  const plain=data.roadmapFor(ids[0]).nodes.find(item=>!item.start),pick=()=>phone.querySelector(`.roadmap-node[data-node="${plain.id}"]`);
  await showAt(ids[0],358);scroller().scrollLeft=7;await press(pick());assert.equal(scroller().scrollLeft,7,'A selection keeps the user\'s scroll');await press(pick());
  viewport=undefined;await act(async()=>phoneRoot.unmount());phone.remove();
  console.log(`PASS collection roadmap phone scroll: on mount and map change an overflowing canvas centers its start node, clamped at both ends, across all ${ids.length} maps at four widths; selections and non-overflowing scrollers untouched`);

  // States: matched nodes follow paperProgress, unmatched are missing, nothing is claimed before progress loads.
  const labels={done:'Solved',started:'In progress',todo:'Not started',missing:'No exercise yet'},states=()=>$$('.roadmap-node').map(element=>element.dataset.state);
  const checkNodes=()=>{for(const item of roadmap.nodes){const element=node(item.id),state=element.dataset.state;assert.equal(element.getAttribute('aria-label'),state==='pending'?`${item.title}, ${item.year}`:`${item.title}, ${item.year}, ${labels[state]}`);assert.deepEqual([...element.children].map(child=>child.className),match(item)&&state!=='pending'?['roadmap-node-title','roadmap-node-bar']:['roadmap-node-title'],item.id);}};
  assert.deepEqual(states(),roadmap.nodes.map(item=>match(item)?'todo':'missing'));checkNodes();
  for(const item of roadmap.nodes.filter(match))assert.equal(node(item.id).querySelector('.roadmap-node-bar > span').style.width,'0%');
  const accepted=module=>({acceptedReport:{problem:module.id,summary:Object.fromEntries(Object.entries(module.tests).map(([layer,count])=>[layer,[count,count]])),tests:Object.entries(module.tests).flatMap(([layer,count])=>Array.from({length:count},(_,index)=>({id:`${layer}-${index}`,layer,passed:true})))}});
  const other=roadmap.nodes.find(item=>match(item)&&item!==ijepa),saved={...Object.fromEntries(paperModules(match(ijepa),problems).map(module=>[module.id,accepted(module)])),[paperModules(match(other),problems)[0].id]:{code:'# edited'}};
  await render({saved});
  assert.equal(node(ijepa.id).dataset.state,'done');assert.equal(node(ijepa.id).querySelector('.roadmap-node-bar > span').style.width,'100%');assert.equal(node(other.id).dataset.state,'started');checkNodes();
  await render({saved,progressReady:false});
  assert.deepEqual(states(),roadmap.nodes.map(item=>match(item)?'pending':'missing'));assert(!$('.roadmap-node-bar'),'No progress bar while progress loads');checkNodes();
  console.log('PASS collection roadmap states: todo/missing, done with full bar, started, pending without progress claims, accessible labels');

  // Selection: lineage stays lit, the rest dims, and the panel reads h3, meta, sentence, the From and To relation sentences (edge
  // order; a side without edges is omitted), the arXiv card, the missing note and the one action.
  const meta=item=>item.venue.includes(String(item.year))?item.venue:`${item.year} · ${item.venue}`;
  const relations=(id,zh)=>[['from',zh?'启发自':'From',roadmap.edges.filter(([,to])=>to===id)],['to',zh?'启发了':'To',roadmap.edges.filter(([from])=>from===id)]].filter(([,,edges])=>edges.length).map(([dir,label,edges])=>[dir,label,edges.map(edge=>edge[zh?2:3])]);
  const shownRelations=()=>[...panel().querySelectorAll('.roadmap-links')].map(section=>{assert(section.localName==='div'&&section.parentElement===panel());assert.deepEqual([...section.children].map(child=>child.localName),['p','ul']);const [label,list]=section.children;assert([...list.children].every(child=>child.localName==='li'));return [section.dataset.dir,label.textContent,[...list.children].map(child=>child.textContent)];});
  const parts=['h3','.roadmap-meta','.roadmap-sentence','.roadmap-links[data-dir="from"]','.roadmap-links[data-dir="to"]','aside.paper-preview-card','.roadmap-missing','a.roadmap-action'],order=()=>[...panel().children].map(child=>parts.find(selector=>child.matches(selector))||child.outerHTML);
  // The arXiv card is the compact community paper card, with the roadmap's local cover and the date as a calendar day.
  const day=(published,zh)=>new Date(published+'T00:00:00Z').toLocaleDateString(zh?'zh-CN':'en-US',{year:'numeric',month:'short',day:'numeric',timeZone:'UTC'});
  const checkCard=(item,zh)=>{
   const cards=panel().querySelectorAll('aside.paper-preview-card');
   if(!item.arxiv)return assert.equal(cards.length,0,item.id+' has no arXiv id and no card');
   const card=cards[0],paper=data.roadmapPapers[item.arxiv],abs='https://arxiv.org/abs/'+item.arxiv,pdf='https://arxiv.org/pdf/'+item.arxiv,title=card?.querySelector('a.paper-preview-title'),time=card?.querySelector('.paper-preview-source time');
   assert.equal(cards.length,1,item.id+' card');assert(card.classList.contains('compact'),'The compact community card');assert.equal(card.querySelector('a.paper-preview-cover').getAttribute('href'),pdf);
   // Links under the cover: the side column holds the cover and then the links, the copy only the source, title and authors.
   const shape=element=>element&&[...element.children].map(child=>child.localName+'.'+child.className);
   assert.deepEqual([shape(card),shape(card.querySelector('.paper-preview-side')),shape(card.querySelector('.paper-preview-copy'))],[['div.paper-preview-side','div.paper-preview-copy'],['a.paper-preview-cover','div.paper-preview-links'],['div.paper-preview-source','a.paper-preview-title','p.paper-preview-authors']],item.id+' card links sit under the cover');
   assert.deepEqual([title.getAttribute('href'),title.textContent],[abs,paper.title]);assert.equal(card.querySelector('.paper-preview-authors').textContent,paper.authors.slice(0,5).join(', ')+(paper.authors.length>5?' et al.':''));
   assert.deepEqual([time?.getAttribute('datetime')?.slice(0,10),time?.textContent],[paper.published,day(paper.published,zh)],item.id+' date is its calendar day in every time zone');
   assert.deepEqual([...card.querySelectorAll('.paper-preview-links a')].map(link=>[link.textContent,link.getAttribute('href')]),[[zh?'论文页面':'Paper page',abs],[zh?'打开 PDF':'Open PDF',pdf]]);
   assert(card.querySelector('.paper-preview-cover img')?.getAttribute('src').endsWith('roadmap-thumbnails/'+item.arxiv+'.webp'),item.id+' local thumbnail');
  };
  const startStates=new Set(),checkPanel=item=>{
   assert(panel(),item.id);assert.equal(panel().querySelector('h3').textContent,item.title);assert.equal(panel().querySelector('p.roadmap-meta').textContent,meta(item));assert.equal(panel().querySelector('p.roadmap-sentence').textContent,item.sentence_en);
   assert.deepEqual(shownRelations(),relations(item.id),item.id+' relation sentences');
   assert.deepEqual(order(),['h3','.roadmap-meta','.roadmap-sentence',...relations(item.id).map(([dir])=>`.roadmap-links[data-dir="${dir}"]`),...item.arxiv?['aside.paper-preview-card']:[],...match(item)?[]:['.roadmap-missing'],'a.roadmap-action'],item.id+' panel order');
   checkCard(item);
   const lit=model.neighbors(roadmap,item.id);assert.deepEqual($$('.roadmap-node:not([data-dimmed])').map(element=>element.dataset.node),roadmap.nodes.filter(entry=>lit.has(entry.id)).map(entry=>entry.id));assert.deepEqual(dimmed('.roadmap-edge'),roadmap.edges.filter(edge=>!edge.includes(item.id)).map(([from,to])=>from+'>'+to));
   // The start tag dims with its node (World Models: lit when it is selected, dimmed for I-JEPA).
   const startDimmed=!lit.has(roadmap.nodes.find(entry=>entry.start).id);assert.equal($('.roadmap-start').hasAttribute('data-dimmed'),startDimmed,item.id+' start tag dims with its node');startStates.add(startDimmed);
   assert.deepEqual($$('.roadmap-node[aria-pressed="true"]').map(element=>element.dataset.node),[item.id]);
   checkLineage(item);
  };
  // Lineage: incoming edges are 'from' (cyan), outgoing 'to' (pink), each with its own arrowhead; one flow overlay per lit edge,
  // after every edge, on the same curve; neighbors carry their side. Exact lists, so nothing from an earlier selection survives.
  const checkLineage=item=>{
   const pairs=edges=>edges.map(([from,to])=>from+'>'+to),incoming=roadmap.edges.filter(([,to])=>to===item.id),outgoing=roadmap.edges.filter(([from])=>from===item.id);
   assert.deepEqual($$('path.roadmap-edge[data-dir="from"]').map(pair),pairs(incoming),item.id+' incoming edges');assert.deepEqual($$('path.roadmap-edge[data-dir="to"]').map(pair),pairs(outgoing),item.id+' outgoing edges');
   assert.deepEqual($$('path.roadmap-edge[data-lit]').map(pair),$$('path.roadmap-edge[data-dir]').map(pair),'Lit edges are exactly the directed ones');
   for(const edge of $$('path.roadmap-edge'))assert.equal(edge.getAttribute('marker-end'),`url(#${base}${edge.dataset.dir?'-'+edge.dataset.dir:''})`,pair(edge)+' arrowhead');
   const lit=$$('path.roadmap-edge[data-dir]'),flows=$$('path.roadmap-flow'),children=[...canvas().querySelector(':scope > svg').children];
   assert.deepEqual(flows.map(flow=>[pair(flow),flow.getAttribute('d'),flow.dataset.dir]),lit.map(edge=>[pair(edge),edge.getAttribute('d'),edge.dataset.dir]),item.id+' flow overlays');
   assert(flows.every(flow=>!flow.hasAttribute('marker-end')&&children.indexOf(flow)>children.indexOf($$('path.roadmap-edge').at(-1))),'Flows have no arrowhead and draw over every edge');
   assert.deepEqual($$('.roadmap-node[data-lineage]').map(element=>[element.dataset.node,element.dataset.lineage]),roadmap.nodes.flatMap(entry=>incoming.some(([from])=>from===entry.id)?[[entry.id,'from']]:outgoing.some(([,to])=>to===entry.id)?[[entry.id,'to']]:[]),item.id+' neighbor sides');
  };
  await render();await press(node(ijepa.id));checkPanel(ijepa);
  let action=panel().querySelector('a.roadmap-action');assert.equal(action.getAttribute('href'),nav.withLibraryContext('papers/i-jepa',returnTo));assert.equal(action.textContent,'Practice');assert(action.querySelector('svg'),'Practice carries the arrow icon');assert(!panel().querySelector('.roadmap-missing'));
  const collection=paperCollections.find(entry=>entry.id==='jepa-world-models'),paperIds=collection.paperIds;
  try{collection.paperIds=paperIds.filter(id=>id!=='i-jepa');await render();assert.equal(panel().querySelector('a.roadmap-action').getAttribute('href'),'#/papers/i-jepa','Papers outside the collection link without collection context');}finally{collection.paperIds=paperIds;await render();}
  await press(node(missing.id));checkPanel(missing);
  assert(roadmap.edges.filter(([from,to])=>[from,to].includes(ijepa.id)&&![from,to].includes(missing.id)).every(([from,to])=>!$(['data-dir','data-lit'].map(name=>`path[data-from="${from}"][data-to="${to}"][${name}]`).join())),'Edges of the previous selection lose their direction, light and flow');
  assert.equal(panel().querySelector('p.roadmap-missing').textContent,'No exercise yet');action=panel().querySelector('a.roadmap-action');assert.equal(action.getAttribute('href'),'#/submit');assert.equal(action.textContent,'Submit this paper');
  // Omitted sides: the missing anchor has no incoming edge, a leaf no outgoing one.
  const leaf=roadmap.nodes.find(item=>!roadmap.edges.some(([from])=>from===item.id)),arxivId=missing.arxiv;
  assert(!roadmap.edges.some(([,to])=>to===missing.id)&&leaf&&leaf!==missing,'The anchors cover both omitted sides');
  await press(node(leaf.id));checkPanel(leaf);
  // The card-less branch: every map node has an arXiv id, so the missing anchor loses its id in the loaded data for one selection.
  try{delete missing.arxiv;await press(node(missing.id));checkPanel(missing);}finally{missing.arxiv=arxivId;await render();}
  assert(panel().querySelector('aside.paper-preview-card'),'The restored arXiv id brings the card back');
  await press(node(ijepa.id));assert(panel().querySelector('.paper-preview-authors').textContent.endsWith(' et al.'),'Six authors show five, then et al.');
  // A cover that fails to load gives way to the PDF placeholder with every link intact; the next paper still shows its own cover.
  // Each paper gets its own card (keyed by arXiv id), so neither a failed state nor the previous cover image carries over.
  const cover=()=>panel().querySelector('a.paper-preview-cover'),card=()=>panel().querySelector('aside.paper-preview-card'),cardLinks=()=>[...panel().querySelectorAll('aside.paper-preview-card a')].map(link=>link.getAttribute('href')),links=cardLinks(),failedCard=card();
  assert.deepEqual(links,['pdf','abs','pdf','abs'].map(kind=>`https://arxiv.org/${kind}/${ijepa.arxiv}`));
  await act(async()=>cover().querySelector('img').dispatchEvent(new dom.window.Event('error')));
  assert(!cover().querySelector('img')&&cover().querySelector('svg')&&cover().querySelector('small')?.textContent==='PDF','The PDF placeholder replaces a cover that failed');assert.deepEqual(cardLinks(),links,'The links survive a failed cover');
  const next=roadmap.nodes.find(item=>item.arxiv&&item!==ijepa);await press(node(next.id));assert(cover().querySelector('img')?.getAttribute('src').endsWith(`roadmap-thumbnails/${next.arxiv}.webp`),'Another paper shows its cover after one failed');
  assert(card()!==failedCard,'The next paper gets a new card, not the one whose cover failed');
  const image=cover().querySelector('img'),third=roadmap.nodes.find(item=>item.arxiv&&item!==ijepa&&item!==next);await press(node(third.id));
  assert(cover().querySelector('img')&&cover().querySelector('img')!==image,'Switching papers swaps in a new image element, so the previous cover never shows while the next loads');
  assert.equal(startStates.size,2,'Selections cover a dimmed and a lit start tag');
  console.log('PASS collection roadmap selection: panel title/meta/sentence, From/To relation sentences in edge order (empty sides omitted), compact arXiv card (local cover with the links under it, metadata, calendar date; placeholder on a failed cover; a new card and image per paper; none once a node loses its arXiv id) before the missing note and action, neighbors lit with their side, incoming/outgoing edges directed with own arrowheads and flow overlays, nothing stale after switching, other nodes, edges and the start tag dimmed, Practice and Submit actions');

  // Keyboard: Escape from anywhere inside returns focus to the node; a second click deselects.
  await press(node(ijepa.id));action=panel().querySelector('a.roadmap-action');action.focus();
  await act(async()=>action.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true})));
  assert(!panel());assert(document.activeElement===node(ijepa.id),'Escape returns focus to the previously selected node');assert.equal(node(ijepa.id).getAttribute('aria-pressed'),'false');assert.deepEqual([...dimmed('.roadmap-node'),...dimmed('.roadmap-edge')],[]);idleEdges();
  await press(node(ijepa.id));assert(panel());await press(node(ijepa.id));assert(!panel(),'Clicking the selected node again clears the selection');idleText();idleEdges();
  console.log('PASS collection roadmap keyboard: Escape clears and restores node focus, repeated click deselects; both leave no flow, direction, lineage or dimming behind');

  // Safari and Firefox on macOS do not focus a clicked button, so keydown reaches the body. Escape there still deselects, but
  // an Escape another handler used, one sent while composing, one sent with a dialog open or from focus outside the roadmap does not.
  const clickOnly=async element=>{assert(element);await act(async()=>{document.activeElement?.blur();element.click();});assert(document.activeElement===document.body,'A click alone leaves focus on the body');};
  const escape=async(init,target=document.body)=>{const event=new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...init});await act(async()=>target.dispatchEvent(event));return event;};
  await clickOnly(node(ijepa.id));assert(panel(),'A click without focus selects');
  const used=event=>event.preventDefault();document.body.addEventListener('keydown',used);await escape();document.body.removeEventListener('keydown',used);assert(panel(),'An Escape another handler already used keeps the selection');
  await escape({isComposing:true});assert(panel(),'Escape during IME composition keeps the selection');
  const dialog=document.body.appendChild(document.createElement('dialog'));dialog.setAttribute('open','');await escape();dialog.remove();assert(panel(),'Escape while a dialog is open keeps the selection');
  const outside=document.body.appendChild(document.createElement('button'));outside.focus();await escape({},outside);outside.remove();assert(panel(),'Escape from focus outside the roadmap keeps the selection');
  document.activeElement?.blur();const sent=await escape();
  assert(!panel(),'Escape sent to the body deselects after a click that did not focus the node');assert(sent.defaultPrevented,'The roadmap claims the Escape it handled');assert(document.activeElement===node(ijepa.id),'Escape moves focus to the node that was selected');assert.equal(node(ijepa.id).getAttribute('aria-pressed'),'false');
  await escape();assert(!panel()&&document.activeElement===node(ijepa.id),'Escape with nothing selected does nothing');
  console.log('PASS collection roadmap Escape without focus: body Escape deselects; used, composing, dialog and outside Escapes are left alone');

  // The selected node names the panel it controls and the sentence that describes it, and dimmed nodes leave the tab order, so
  // Tab goes from the selected node through its lit neighbors to the panel action.
  const tabbable=()=>[...$('.collection-roadmap').querySelectorAll('button,a[href]')].filter(element=>element.tabIndex>=0).map(element=>element.dataset.node||element.getAttribute('href'));
  assert.deepEqual(tabbable(),roadmap.nodes.map(item=>item.id),'Every node is in the tab order when nothing is selected');
  for(const item of [ijepa,missing]){
   await press(node(item.id));
   const aside=panel(),heading=aside.querySelector('h3'),sentence=aside.querySelector('p.roadmap-sentence'),lit=model.neighbors(roadmap,item.id),order=tabbable();
   assert(aside.id&&heading.id&&sentence.id,'Panel, heading and sentence have ids');assert.equal(aside.getAttribute('aria-labelledby'),heading.id,'The panel is named by its heading');
   assert.equal(node(item.id).getAttribute('aria-controls'),aside.id);assert.equal(node(item.id).getAttribute('aria-describedby'),sentence.id);
   assert.deepEqual($$('.roadmap-node[aria-controls],.roadmap-node[aria-describedby]').map(element=>element.dataset.node),[item.id],'Only the selected node points at the panel');
   assert.deepEqual($$('.roadmap-node').filter(element=>element.tabIndex<0).map(element=>element.dataset.node),roadmap.nodes.filter(entry=>!lit.has(entry.id)).map(entry=>entry.id),'Dimmed nodes leave the tab order');
   assert.deepEqual(order.slice(order.indexOf(item.id)+1),[...roadmap.nodes.slice(roadmap.nodes.indexOf(item)+1).filter(entry=>lit.has(entry.id)).map(entry=>entry.id),...['pdf','abs','pdf','abs'].map(kind=>`https://arxiv.org/${kind}/${item.arxiv}`),panel().querySelector('a.roadmap-action').getAttribute('href')],'Tab goes from the lit neighbors through the card links (cover, Paper page, Open PDF, title) to the panel action');
  }
  await press(node(missing.id));assert(!panel());assert.deepEqual(tabbable(),roadmap.nodes.map(item=>item.id),'Deselecting returns every node to the tab order');
  console.log('PASS collection roadmap panel access: aria-controls/-describedby on the selected node, labelled panel, dimmed nodes out of the tab order');

  await press(node(ijepa.id));await act(async()=>setLocale('zh'));
  assert.equal(panel().querySelector('p.roadmap-sentence').textContent,ijepa.sentence);assert.equal($('.roadmap-start').textContent,'从这里开始');assert.equal(canvas().getAttribute('aria-label'),'路线图');assert.equal(panel().querySelector('a.roadmap-action').textContent,'练习');assert.equal(node(missing.id).getAttribute('aria-label'),`${missing.title}, ${missing.year}, 暂无练习`);
  assert.deepEqual(shownRelations(),relations(ijepa.id,true),'zh relation sentences under zh labels');checkCard(ijepa,true);
  await act(async()=>setLocale('en'));assert.deepEqual(shownRelations(),relations(ijepa.id),'Back to en relation sentences');checkCard(ijepa);
  await render({collectionId:'nope'});assert(!$('.collection-roadmap'),'A collection without roadmap data renders nothing');
  console.log('PASS collection roadmap locale: zh sentence, relation sentences and card labels and date, start tag, canvas label and action, all switching with a node selected; unknown collection renders nothing');

  process.env.VITE_PUBLIC_SITE='1';delete process.env.VITE_HOSTED_APP_AVAILABLE;
  ui=await load(await serve());assert.equal(ui.deployment.publicSite,true,'Vite reads VITE_PUBLIC_SITE at server creation');assert.equal(ui.deployment.hostedAppAvailable,false);
  await render();await press(node(missing.id));assert.equal(panel().querySelector('p.roadmap-missing').textContent,'No exercise yet');assert(!panel().querySelector('.roadmap-action'),'No submit link where the site has no submit page');
  await press(node(ijepa.id));assert.equal(panel().querySelector('a.roadmap-action').getAttribute('href'),nav.withLibraryContext('papers/i-jepa',returnTo));
  assert.deepEqual(errors,[],'No jsdom errors');assert.deepEqual(consoleErrors,[],'No React warnings');
  console.log('PASS collection roadmap editions: public site without hosted app hides Submit, keeps Practice; no DOM errors or React warnings');
 }finally{if(root)await act(async()=>root.unmount());for(const{server,cacheDir}of servers){await server.close();await rm(cacheDir,{recursive:true,force:true});}dom.window.close();for(const[key,value]of original)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];for(const[key,value]of Object.entries(env))if(value===undefined)delete process.env[key];else process.env[key]=value;console.error=consoleError;}
}

// Section: roadmap styles. jsdom has no layout, so these pin the rules behind the wrapping sticky panel, node title room and forced colors.
{
 const css=normalizeCss(await readFile('src/collection-roadmap.css','utf8')),every=outsideMedia(css),has=(list,...want)=>want.every(item=>list.includes(item)),weight=selector=>(selector.match(/\.[\w-]+|\[[^\]]+\]|:[\w-]+/g)||[]).length;
 assert(has(decls(every,'.collection-roadmap'),'display:flex','flex-wrap:wrap'),'The panel wraps below the canvas instead of squeezing it');
 assert(has(decls(every,'.roadmap-canvas-scroll'),'max-width:100%','overflow-x:auto'),'The canvas scroller never grows past the column');
 assert(has(decls(every,'.roadmap-panel'),'position:sticky','top:calc(var(--topbar-offset,56px) + 16px)'),'Beside the canvas the panel stays in view 16px below the site header, level with the sticky overview sidebar');
 assert(!decls(css,'.roadmap-panel').some(item=>/^(bottom|inset)/.test(item)),'No bottom offset at any width: a wrapped panel (up to ~890px tall with the card) stays in flow under the canvas instead of floating over the graph');
 assert(![...css.matchAll(/@media[^{]*\{((?:[^{}]*\{[^{}]*\})*)\}/g)].some(([,body])=>/(^|\})\.roadmap-panel\{[^}]*top:/.test(body)),'No width-specific offset: --topbar-offset already follows the one- and two-row header');
 assert(has(decls(media(css,'@media (max-height:600px)'),'.roadmap-panel'),'position:static'),'Short or zoomed viewports keep the panel in flow so it never hides the focused node');
 assert(has(decls(media(css,'@media (max-width:850px)'),'.roadmap-panel'),'flex:1 1 100%','max-width:none'),'On narrow screens the wrapped panel spans the column under the canvas');
 assert(has(decls(every,'.roadmap-panel'),'flex:1 1 200px','max-width:320px','min-width:0','container-type:inline-size'),'Beside the canvas the panel takes the 200-320px left over, so it stays beside a 674px canvas down to an 890px column (912px at a 1280px viewport)');
 // The compact community card with its links under the cover, roadmap-scoped (community pages keep theirs): an 84px cover leaves
 // the title 4 lines at meta size and the authors 3. A panel under 260px (content box under 226px: 2x16px padding and 2x1px
 // border) stacks the card and turns the cover column into a row, links right of the cover; those rules have the same selectors
 // and come later, so they win when narrow. The side column's base styles (a column, links stacked without a top margin) are the card's.
 const card='.roadmap-panel .paper-preview-card.compact',stacked='@container (max-width:225px)',community=normalizeCss(await readFile('src/community-papers.css','utf8')),base=outsideMedia(community);
 assert(has(decls(base,'.paper-preview-side'),'display:flex','flex-direction:column')&&has(decls(base,'.paper-preview-side .paper-preview-links'),'flex-direction:column','margin-top:0'),'The card stacks its cover and links in a side column, one link per line');
 // Under an 84px cover the labels never wrap: caption size and a 3px icon gap fit "Paper page" (90px at meta size, 84px this way).
 assert(has(decls(base,'.paper-preview-side .paper-preview-links a'),'gap:3px','font-size:var(--type-caption)','white-space:nowrap'),'Links under the cover stay on one line each');
 assert(has(decls(every,card),'grid-template-columns:84px minmax(0,1fr)','gap:10px','padding:10px')&&has(decls(every,card+' .paper-preview-cover'),'width:84px'),'An 84px cover beside the card text');
 assert(has(decls(every,card+' .paper-preview-title'),'font-size:var(--type-meta)','-webkit-line-clamp:4')&&has(decls(every,card+' .paper-preview-authors'),'-webkit-line-clamp:3'),'Card titles get up to 4 lines at meta size, authors up to 3');
 assert(has(decls(media(css,stacked),card),'grid-template-columns:1fr')&&has(decls(media(css,stacked),card+' .paper-preview-side'),'flex-direction:row')&&css.indexOf(stacked)>css.indexOf(card+'{'),'A panel under 260px stacks the card with the links right of the cover, after the 84px rules so it wins');
 // No community rule on the card, its side column, cover, title or authors (any width) is as specific as the roadmap override for that element.
 // Rules scoped to a community feed row (.community-post-card) never reach the roadmap panel, so they are not rivals.
 const target=selector=>selector.split(/[\s>+~]+/).at(-1).match(/^\.paper-preview-(card|side|cover|title|authors)(?![\w-])/)?.[0],feedRow=selector=>/^\.community-post-card[\s>]/.test(selector.trim());
 for(const [,list] of community.matchAll(/([^{}]+)\{[^{}]*\}/g))for(const selector of list.split(',')){const element=!feedRow(selector)&&target(selector);if(element)assert(weight(selector)<weight(element==='.paper-preview-card'?card:card+' '+element),selector+' would beat the roadmap card override');}
 // Relation sentences: plain lists whose bullets are dots in the direction colors (cyan From, pink To).
 assert(has(decls(every,'.roadmap-links>ul'),'list-style:none','padding:0'),'Relation lists drop the default bullets');
 assert(has(decls(every,'.roadmap-links li::before'),'content:""','border-radius:50%'),'Each relation sentence starts with a dot');
 for(const dir of ['from','to'])assert(has(decls(every,`.roadmap-links[data-dir="${dir}"] li::before`),`background:var(--roadmap-${dir})`),dir+' dots take the direction color');
 // From and To labels take text tokens of the direction colors: the edge colors in dark, darker tones in light so the caption-size
 // labels reach 4.5:1 on the panel surface (--surface: #262626 dark, #fff light). Forced colors repaint text on their own.
 assert(has(decls(every,'.collection-roadmap'),'--roadmap-from-text:#22d3ee','--roadmap-to-text:#f472b6')&&has(decls(every,'[data-theme="light"] .collection-roadmap'),'--roadmap-from-text:#0e7490','--roadmap-to-text:#be185d'),'From/To label tokens per theme');
 for(const dir of ['from','to'])assert(has(decls(every,`.roadmap-links[data-dir="${dir}"]>p`),`color:var(--roadmap-${dir}-text)`),dir+' label takes its direction text color');
 const luminance=hex=>[1,3,5].map(at=>parseInt(hex.slice(at,at+2),16)/255).map(c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4).reduce((sum,c,i)=>sum+c*[.2126,.7152,.0722][i],0),contrast=(a,b)=>{const [x,y]=[luminance(a),luminance(b)].sort((m,n)=>n-m);return (x+.05)/(y+.05);};
 for(const [text,surface] of [['#22d3ee','#262626'],['#f472b6','#262626'],['#0e7490','#ffffff'],['#be185d','#ffffff']])assert(contrast(text,surface)>=4.5,`${text} on ${surface} reaches AA`);
 assert(has(decls(every,'.roadmap-node'),'padding:6px'),'Node titles get 6px side padding (110px of text)');
 assert(has(decls(every,'.roadmap-node[data-dimmed]:focus-visible'),'opacity:1'),'A dimmed node with keyboard focus is fully visible');
 assert(has(decls(every,'.roadmap-node[data-dimmed],.roadmap-edge[data-dimmed],.roadmap-start[data-dimmed]'),'opacity:.25')&&has(decls(every,'.roadmap-start'),'transition:opacity .16s')&&has(decls(media(css,'@media (prefers-reduced-motion:reduce)'),'.roadmap-node,.roadmap-edge,.roadmap-start'),'transition:none'),'The start tag fades with its node, as far and as fast (instantly under reduced motion)');
 const forced=media(css,'@media (forced-colors:active)');
 assert(has(decls(forced,'.roadmap-node[data-state]'),'border-color:CanvasText')&&has(decls(forced,'.roadmap-node[data-state="missing"]'),'border-color:GrayText'),'Forced colors keep the state borders');
 assert(has(decls(forced,'.roadmap-node-bar'),'outline:1px solid CanvasText')&&has(decls(forced,'.roadmap-node-bar>span'),'forced-color-adjust:none','background:CanvasText'),'Forced colors keep the progress bar outline and fill');
 assert(has(decls(forced,'.roadmap-node[aria-pressed="true"]'),'outline:3px solid Highlight'),'Forced colors keep the selection ring');
 assert(has(decls(forced,'.roadmap-links[data-dir] li::before'),'forced-color-adjust:none','background:CanvasText'),'Forced colors keep the relation dots (as specific as the direction rules, and later)');
 // Lineage colors: tokens per theme, direction-colored lit edges and arrowheads, lighter dashes flowing downstream that stop for
 // reduced motion and forced colors; idle edges and missing nodes stay neutral.
 assert(has(decls(every,'.collection-roadmap'),'--roadmap-from:#22d3ee','--roadmap-to:#f472b6','--roadmap-glow:#a78bfa','--roadmap-started:#fbbf24'),'Dark (default) lineage tokens');
 assert(has(decls(every,'[data-theme="light"] .collection-roadmap'),'--roadmap-from:#0891b2','--roadmap-to:#db2777','--roadmap-glow:#7c3aed','--roadmap-started:#d97706'),'Light lineage tokens');
 assert(has(decls(every,'.roadmap-edge'),'stroke:var(--text-secondary)')&&has(decls(every,'.roadmap-node[data-state="missing"]'),'border:1.5px dashed color-mix(in srgb,var(--text-secondary) 55%,transparent)'),'Idle edges and missing nodes stay neutral');
 assert(has(decls(every,'.roadmap-edge[data-lit]'),'stroke-opacity:1','stroke-width:2.5'),'Lit edges are 2.5px');
 for(const dir of ['from','to'])assert(has(decls(every,`.roadmap-edge[data-dir="${dir}"]`),`stroke:var(--roadmap-${dir})`)&&has(decls(every,`.roadmap-arrow.${dir} path`),`fill:var(--roadmap-${dir})`)&&has(decls(every,`.roadmap-flow[data-dir="${dir}"]`),`stroke:color-mix(in srgb,var(--roadmap-${dir}) 35%,#fff)`),dir+' edges, arrowheads and lighter flow dashes take the direction color');
 assert(has(decls(every,'.roadmap-flow'),'stroke-dasharray:2 10','animation:roadmap-flow .9s linear infinite'),'Short dashes flow along lit edges');
 assert(has(decls(media(css,'@keyframes roadmap-flow'),'to'),'stroke-dashoffset:-24'),'Dashes travel downstream, two dash cycles per loop');
 assert(has(decls(media(css,'@media (prefers-reduced-motion:reduce)'),'.roadmap-flow'),'display:none'),'Reduced motion keeps the colors without the flow');
 assert(has(decls(forced,'.roadmap-flow'),'display:none')&&has(decls(forced,'.roadmap-edge[data-dir]'),'stroke:Highlight')&&has(decls(forced,'.roadmap-arrow.from path,.roadmap-arrow.to path'),'fill:Highlight'),'Forced colors drop the flow and draw lit edges in Highlight');
 // The selected node: cyan (top) to pink (bottom) border and a violet glow. Hover only tints the fill color, and every other node
 // rule that paints a border, background or shadow loses to it on specificity or order (solved, in progress, missing, neighbor).
 const pressed='.roadmap-node[aria-pressed="true"]';
 assert(decls(every,pressed).some(item=>item.startsWith('background:')&&item.includes('linear-gradient(var(--roadmap-from),var(--roadmap-to)) border-box'))&&has(decls(every,pressed),'border:2px solid transparent'),'The selected border is a 2px cyan-to-pink gradient, also on missing (1.5px dashed) papers');
 assert(decls(every,pressed).some(item=>item.startsWith('box-shadow:')&&item.includes('var(--roadmap-glow)')),'The selected node glows violet');
 const unforced=css.replace('@media (forced-colors:active){'+forced+'}',''),rules=[...unforced.matchAll(/([^{}]+)\{([^{}]*)\}/g)].flatMap(match=>match[1].split(',').map(selector=>({selector:selector.trim(),at:match.index,body:match[2].split(';')})));
 const onNode=rule=>/^\.roadmap-node(?![\w-])[^\s>+~]*$/.test(rule.selector),hovers=rules.filter(rule=>onNode(rule)&&rule.selector.includes(':hover'));assert(hovers.length&&hovers.every(rule=>rule.body.every(item=>item.startsWith('background-color:'))),'Hover only changes the fill color, under the selected gradient');
 const selectedAt=rules.find(rule=>rule.selector===pressed).at,rivals=rules.filter(rule=>onNode(rule)&&rule.selector!==pressed&&rule.body.some(item=>/^(background|background-image|border|border-color|border-style|border-width|box-shadow):/.test(item)));
 assert(['[data-state="done"]','[data-state="started"]','[data-state="missing"]','[data-lineage="from"]','[data-lineage="to"]'].every(state=>rivals.some(rule=>rule.selector==='.roadmap-node'+state)),'Every state and lineage border rule is compared');
 for(const rule of rivals)assert(weight(rule.selector)<weight(pressed)||weight(rule.selector)===weight(pressed)&&rule.at<selectedAt,rule.selector+' would repaint the selected border');
 assert(has(decls(every,'.roadmap-node[data-state="started"]'),'border-color:var(--roadmap-started)')&&has(decls(every,'.roadmap-node-bar>span'),'background:var(--roadmap-started)'),'In progress is yellow');
 for(const dir of ['from','to'])assert(has(decls(every,`.roadmap-node[data-lineage="${dir}"]`),`border-color:var(--roadmap-${dir})`),dir+' neighbors take the direction color');
 assert(has(decls(every,'.roadmap-start'),'background:linear-gradient(90deg,#f87171,#fb923c,#facc15,#4ade80,#22d3ee,#60a5fa,#a78bfa)','color:#111827','transform:translate(-50%,calc(-100% - 6px))'),'Spectrum start tag, 6px (START_TAG.gap) above its node');
 assert(has(decls(every,'.roadmap-start'),'padding:1px 10px','line-height:20px'),'The tag is exactly START_TAG.height (22px) tall, so the clearance box in the data rules is never smaller than the tag');
 assert(has(decls(every,'.roadmap-start[data-side="left"]'),'transform:translate(-100%,-50%)'),'A left tag ends at its anchor (START_TAG.side before the node), vertically centered');
 assert(!/var\(--(accent|chart-1)\)/.test(css),'No site accent (grey-and-orange highlight) remains');
 console.log('PASS collection roadmap styles: 200-320px panel sticky beside the canvas and full width in flow once wrapped under it, scroller capped at the column, roadmap-scoped card (84px cover with the links under it, 4-line title, 3-line authors, stacked under 260px with the links beside the cover, beating every community rule), relation lists with direction-colored dots and AA From/To labels, 6px node padding, focused dimmed nodes, start tag dimmed with its node, forced-colors progress, borders and dots; lineage tokens in both themes, direction-colored lit edges with a flowing overlay (off for reduced motion and forced colors), gradient selected border that hover, state and lineage rules cannot override, yellow progress, spectrum start tag above or left');
}

// Section: collection page (Papers / Roadmap view tabs and the lazy view inside PaperLibrary)
{
 const errors=[],consoleErrors=[],consoleError=console.error,virtualConsole=new VirtualConsole();virtualConsole.on('jsdomError',error=>errors.push(error));console.error=(...args)=>consoleErrors.push(args.join(' '));
 const home='#/collections/jepa-world-models',roadmapHref=home+'?view=roadmap',dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/'+home,pretendToBeVisual:true,virtualConsole});
 const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,location:dom.window.location,history:dom.window.history,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,innerHeight:900,scrollY:0,scrollTo(){},ResizeObserver:class{observe(){}disconnect(){}},requestAnimationFrame:callback=>setTimeout(callback,0),cancelAnimationFrame:clearTimeout,IS_REACT_ACT_ENVIRONMENT:true};
 const original=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
 window.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});window.scrollTo=()=>{};window.HTMLElement.prototype.scrollIntoView=function(){};localStorage.setItem('paperleet.locale','en');
 // failRoadmap makes the lazy roadmap module fail to load, as an offline or stale chunk would.
 let failRoadmap=true;
 const cacheDir=await mkdtemp(tmpdir()+'/paperleet-collection-roadmap-page-'),server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'roadmap-chunk-failure',load(id){if(failRoadmap&&id.endsWith('/src/CollectionRoadmap.tsx'))throw new Error('Simulated roadmap chunk failure');}}]});let root;
 try{
  const catalog=JSON.parse(await readFile('public/data/catalog.json','utf8')),{createRoot}=await import('react-dom/client'),{renderToString}=await import('react-dom/server'),{PaperLibrary}=await server.ssrLoadModule('/src/PaperViews.tsx'),{paperCollections}=await server.ssrLoadModule('/src/paperCollections.ts'),{LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx'),nav=await server.ssrLoadModule('/src/libraryNavigation.ts'),h=React.createElement;
  // Course rows (company tools) and the global library (featured collections) defer these modules; loading them up front leaves settle() only cached chunks to wait for.
  for(const module of ['/src/LibraryCompanyEvidence.tsx','/src/LibraryCollections.tsx'])await server.ssrLoadModule(module);
  // The Frame of check-library-filter-continuity, plus the paging flag and a stack of pushed (Back-able) entries.
  let route=home,last;const pushed=[];
  function Frame(){const view=nav.readLibraryRoute(route);return h(PaperLibrary,{...catalog,route,collectionId:view?.collectionId,saved:{},onFilters:(href,paging)=>{last={href,paging};if(paging)pushed.push(href);route=href;history.replaceState(null,'',href);root.render(tree());}});}
  const tree=()=>h(React.StrictMode,null,h(LocaleProvider,null,h(Frame)));
  const render=async href=>act(async()=>{route=href;history.replaceState(null,'',href);root??=createRoot(document.getElementById('root'));root.render(tree());});
  const wait=async(predicate,message)=>{for(let n=0;n<180&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),message);};
  const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});};
  // Each deferred row tool lazy-loads per mount and renders nothing in this edition, so no element marks it loaded; a few act-wrapped turns let it resolve.
  const settle=()=>act(async()=>{for(let n=0;n<10;n++)await new Promise(resolve=>setTimeout(resolve,0));});
  const $=selector=>document.querySelector(selector),tabs=()=>[...document.querySelectorAll('.library-heading .collection-views a')],tab=name=>tabs().find(link=>link.textContent===name),heading=()=>$('.library-heading h2'),rows=()=>!!$('[data-paper-id]'),shown=()=>!!$('.collection-roadmap');
  const slash=async()=>{const event=new dom.window.KeyboardEvent('keydown',{key:'/',bubbles:true,cancelable:true});await act(async()=>document.body.dispatchEvent(event));return event;};

  // A roadmap chunk that cannot load shows the shared failure state (alert, Retry, Reload) in place of the list, and Retry recovers.
  await render(roadmapHref);await wait(()=>$('.feature-failure'),'The failure state renders');
  assert.equal($('.feature-failure [role="alert"]').textContent,'Roadmap could not load.');assert.deepEqual([...$('.feature-failure').querySelectorAll('button')].map(button=>button.textContent),['Retry loading','Reload page']);assert(!$('.library-toolbar')&&!shown(),'The failure state replaces the list');
  assert(consoleErrors.length&&consoleErrors.every(message=>message.includes('Simulated roadmap chunk failure')),'Only the simulated failure is reported');consoleErrors.length=0;
  failRoadmap=false;for(const mod of server.moduleGraph.getModulesByFile(new URL('../src/CollectionRoadmap.tsx',import.meta.url).pathname)||[])server.moduleGraph.invalidateModule(mod);await click($('.feature-failure button'));await wait(shown,'Retry loads the roadmap');assert(!$('.feature-failure'));
  console.log('PASS collection roadmap page load failure: alert names the roadmap, Retry and Reload, Retry recovers');

  await render(home);await wait(rows,'The collection page defaults to the paper list');assert(!shown());
  // A collection with a roadmap titles its section with two view tabs; a hidden heading names the current view for screen readers.
  const views=()=>tabs().map(link=>[link.textContent,link.getAttribute('href'),link.getAttribute('aria-current')]);
  assert.deepEqual(views(),[['Papers',home,'page'],['Roadmap',roadmapHref,null]],'Papers and Roadmap tabs, Papers current');
  assert.equal($('.library-heading .collection-views').getAttribute('aria-label'),'Collection views');assert(tabs().every(link=>link.querySelectorAll('svg[aria-hidden="true"]').length===1),'Each tab has one decorative icon');
  assert(heading().matches('.sr-only')&&heading().textContent==='Papers','A hidden heading names the list');
  const row=[...$('.paper-categories').children];
  assert(row[0].matches('button:not([class])')&&row[0].querySelector('[lang="en"]').textContent==='All','All comes first');assert.equal(row[0].getAttribute('aria-pressed'),'true');
  assert(row.length>1&&row.slice(1).every(node=>node.matches('button[class^="category-"]')),'Only category chips follow All');
  let event=await slash();assert(event.defaultPrevented&&document.activeElement===$('.library-toolbar .search input'),'/ focuses search in the list');
  // A modified click is left to the browser, which opens the view in a new tab. The document listener runs after React's root
  // listener, records whether the tab prevented the default, then stops jsdom from following the link in place.
  let nativeDefault;const record=clicked=>{nativeDefault=!clicked.defaultPrevented;clicked.preventDefault();};document.addEventListener('click',record);
  await act(async()=>tab('Roadmap').dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,metaKey:true})));document.removeEventListener('click',record);
  assert(nativeDefault===true&&!last&&!pushed.length,'A modified click does not switch the view in place');

  let plainPrevented;const recordPlain=clicked=>{plainPrevented=clicked.defaultPrevented;};document.addEventListener('click',recordPlain);
  await click(tab('Roadmap'));document.removeEventListener('click',recordPlain);assert(plainPrevented,'A plain click switches in place instead of following the link');
  assert.deepEqual(last,{href:roadmapHref,paging:true});assert.deepEqual(pushed,[roadmapHref]);
  assert.deepEqual(views(),[['Papers',home,null],['Roadmap',roadmapHref,'page']],'Roadmap becomes the current tab');assert.equal(heading().textContent,'Roadmap');
  await wait(shown,'The lazy roadmap renders');
  for(const selector of ['.library-category-browser','.library-toolbar','.problem-rows','.library-results'])assert(!$(selector),selector+' is replaced by the roadmap');
  assert($('.library-main > .library-heading .collection-views')&&$('.library-main .collection-roadmap'),'Tabs and roadmap share the library column');
  await wait(()=>$('.library-difficulty-list button.difficulty-summary.E'),'The overview sidebar stays mounted');
  await click(tab('Roadmap'));assert.deepEqual(pushed,[roadmapHref],'Clicking the current tab does nothing');
  tab('Roadmap').focus();event=await slash();assert(document.activeElement===tab('Roadmap')&&!event.defaultPrevented,'/ is inert in the roadmap');
  console.log('PASS collection roadmap page tabs: Papers and Roadmap links with a hidden heading; categories follow All; modified clicks stay native; pushed view=roadmap; the roadmap replaces chips, toolbar and rows; / inert');

  await click(tab('Papers'));assert.deepEqual(last,{href:home,paging:true});assert.deepEqual(pushed,[roadmapHref,home],'Back from the list returns to the roadmap');
  await wait(()=>rows()&&!shown(),'Rows return after the Papers tab');assert.deepEqual(views(),[['Papers',home,'page'],['Roadmap',roadmapHref,null]]);assert.equal(heading().textContent,'Papers');
  // The sidebar disables levels without modules, so the first enabled level is used.
  const level=()=>$('.library-difficulty-list button.difficulty-summary:not([disabled])');
  await render(roadmapHref);await wait(()=>shown()&&level());const difficulty=level().classList[1];await click(level());
  assert.deepEqual(nav.readLibraryRoute(route)?.filters,{...nav.emptyLibraryFilters(),difficulty},'The sidebar leaves the roadmap for the filtered list');await wait(()=>$('.library-toolbar')&&!shown(),'The list returns after a sidebar filter');
  assert.deepEqual(last,{href:route,paging:true},'Leaving through the sidebar pushes, so Back returns to the roadmap');assert.equal(pushed.at(-1),route);
  await click($('.library-difficulty-list button.difficulty-summary.'+difficulty));assert.equal(last.paging,false,'Sidebar filters inside the list still replace the entry');
  console.log('PASS collection roadmap page exits: the Papers tab and the sidebar difficulty filter push the list; list filters still replace');

  await render('#/collections/llm-basics');await wait(()=>$('[data-module-id]'),'Course page renders');await settle();assert(!tabs().length&&!heading().matches('.sr-only')&&heading().textContent==='Course modules','A course keeps its visible heading and has no view tabs');
  await render('#');await wait(rows,'Global library renders');await settle();assert(!tabs().length&&$('.library-heading h1'),'The global library has no view tabs');
  // With the Roadmap out of the chip row, every collection shows three collapsed category slots, and Expand only when more exist.
  const categoryCount=id=>{const listed=catalog.papers.filter(paper=>paperCollections.find(entry=>entry.id===id).paperIds.includes(paper.id));return catalog.categories.filter(item=>listed.some(paper=>paper.category===item.id)).length;};
  const categoryChips=()=>document.querySelectorAll('.paper-categories button[class^="category-"]').length,toggle=()=>$('.library-category-toggle');
  assert(paperCollections.some(entry=>entry.roadmap&&categoryCount(entry.id)>=3),'A roadmap collection with three or more categories exists');
  for(const entry of paperCollections.filter(entry=>entry.kind!=='course')){
   await render('#/collections/'+entry.id);await wait(rows,entry.id+' renders');
   assert.deepEqual(tabs().map(link=>link.textContent),['Papers','Roadmap'],entry.id+': every featured collection has Papers and Roadmap tabs');
   assert.equal(categoryChips(),Math.min(3,categoryCount(entry.id)),entry.id+': three collapsed category slots');assert.equal(!!toggle(),categoryCount(entry.id)>3,entry.id+': Expand only when more categories exist');
  }
  // A featured collection without a curated roadmap yet (rl, unmarked for this fixture; PaperViews shares this module instance) keeps
  // both tabs; its Roadmap view is a short labelled note in place of the chips and list, loads no roadmap code, and the Papers tab leads back.
  const plain=paperCollections.find(entry=>entry.id==='rl'),plainHome='#/collections/rl',pending=()=>$('.roadmap-pending');delete plain.roadmap;
  try{
   await render(plainHome);await wait(rows,'A collection without a roadmap renders its list');pushed.length=0;
   await click(tab('Roadmap'));assert.deepEqual(last,{href:plainHome+'?view=roadmap',paging:true},'Its Roadmap tab opens the roadmap view');
   assert(pending(),'The roadmap view shows the pending note');assert.equal(pending().querySelector('h3').textContent,'Roadmap in progress');
   const label=pending().getAttribute('aria-labelledby');assert(label&&document.getElementById(label)===pending().querySelector('h3'),'The note is a region named by its heading');
   assert.equal(pending().querySelector('p').textContent,'The key papers for this collection are still being mapped. Start with the paper list for now.');
   for(const selector of ['.collection-roadmap','.collection-roadmap-placeholder','.feature-failure','.library-category-browser','.library-toolbar','.problem-rows'])assert(!$(selector),selector+' is absent while the roadmap is pending');
   assert.equal(heading().textContent,'Roadmap');assert.deepEqual(views().map(([name,,current])=>[name,current]),[['Papers',null],['Roadmap','page']]);
   await click(tab('Papers'));assert.deepEqual(last,{href:plainHome,paging:true},'The Papers tab returns to the list as a new history entry');await wait(()=>rows()&&!pending(),'The list returns');
   assert.equal(document.activeElement,tab('Papers'),'Focus stays on the Papers tab, which outlives the switch');
   // The note in Chinese: the locale is read at mount, so remount under zh, then return to en for the sections below.
   const remount=async locale=>{localStorage.setItem('paperleet.locale',locale);await act(async()=>root.unmount());root=undefined;};
   try{await remount('zh');await render(plainHome+'?view=roadmap');await wait(pending,'The zh note renders');assert.deepEqual([pending().querySelector('h3').textContent,pending().querySelector('p').textContent],['路线图整理中','关键论文还在整理中，可以先从论文列表开始。']);}
   finally{await remount('en');}
  }finally{plain.roadmap=true;}
  const html=renderToString(h(LocaleProvider,null,h(PaperLibrary,{...catalog,route:roadmapHref,collectionId:'jepa-world-models',saved:{}})));
  assert(html.includes('<div class="collection-roadmap-placeholder" aria-hidden="true">')&&!html.includes('library-toolbar')&&!html.includes('library-category-browser'),'Before its chunk loads, the roadmap holds a sized placeholder in place of the chips and list');
  const css=normalizeCss(await readFile('src/library-workbench.css','utf8'));
  const declares=(list,...want)=>want.every(item=>list.includes(item)),base=outsideMedia(css),forced=media(css,'@media (forced-colors:active)');
  const strip='.paper-library .collection-views',tabLink=strip+' a',currentTab=tabLink+'[aria-current="page"]';
  assert(declares(decls(base,strip),'border-bottom:1px solid var(--line)')&&declares(decls(base,tabLink),'min-height:44px','margin-bottom:-1px','border-bottom:2px solid transparent','color:var(--text-secondary)'),'Each 44px tab reserves an underline that sits on the strip hairline, inactive tabs in the secondary text color');
  assert(declares(decls(base,currentTab),'color:var(--text)','border-bottom-color:var(--text)'),'The current tab is underlined in the text color');
  assert(declares(decls(forced,currentTab),'border-bottom-color:Highlight')&&declares(decls(forced,tabLink+':not([aria-current="page"])'),'border-bottom-width:1px','border-bottom-color:CanvasText'),'In forced colors only the current tab keeps a 2px Highlight underline; inactive tabs merge into the hairline');
  assert(css.includes('.paper-library .collection-roadmap-placeholder{min-height:740px}'),'The placeholder is about as tall as a roadmap (650-910px)');
  // The chip row never wraps or overflows the page: library-filters.css scrolls it at every width, ahead of the workbench rules.
  const filters=outsideMedia(normalizeCss(await readFile('src/library-filters.css','utf8'))),chipRow='.paper-library .library-category-browser .paper-categories';
  assert(['flex-wrap:nowrap','overflow-x:auto','min-width:0'].every(rule=>decls(filters,chipRow).includes(rule))&&decls(filters,chipRow+' button').includes('flex-shrink:0'),'At every width the chip row scrolls instead of overflowing');
  assert.deepEqual(errors,[],'No jsdom errors');assert.deepEqual(consoleErrors,[],'No React warnings');
  console.log('PASS collection roadmap page scope: three collapsed category slots everywhere; tabs on every featured collection except a course, with a labelled in-progress note (en and zh) where no roadmap is curated and the Papers tab keeping focus on the way back; no tabs on a course or the global library; SSR placeholder; tab, placeholder and chip-row CSS; no DOM errors or React warnings');
 }finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of original)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];console.error=consoleError;}
}
