import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createServer} from 'vite';

const web=process.cwd(),fixture=await fs.mkdtemp(path.join(os.tmpdir(),'paperleet-worker-watch-'));
let server;
try {
  for(const directory of ['src','public/browser-judge','scripts'])await fs.mkdir(path.join(fixture,directory),{recursive:true});
  await fs.symlink(path.join(web,'node_modules'),path.join(fixture,'node_modules'),'dir');
  await fs.copyFile('scripts/prepare-browser-worker.mjs',path.join(fixture,'scripts/prepare-browser-worker.mjs'));
  await fs.writeFile(path.join(fixture,'src/browserWorker.mjs'),"import {syntaxCheck} from '../public/browser-judge/syntax.mjs'; self.onmessage=()=>self.postMessage(syntaxCheck);\n");
  const helper=path.join(fixture,'public/browser-judge/syntax.mjs');
  await fs.writeFile(helper,"export const syntaxCheck='first version';\n");
  execFileSync(process.execPath,['scripts/prepare-browser-worker.mjs'],{cwd:fixture,stdio:'pipe'});
  const generated=path.join(fixture,'src/browserWorkerAsset.generated.ts');
  const before=await fs.readFile(generated,'utf8');
  server=await createServer({root:fixture,configFile:path.join(web,'vite.config.ts'),server:{middlewareMode:true},
    optimizeDeps:{noDiscovery:true,entries:[]},logLevel:'silent'});
  let reloaded=false;
  const send=server.ws.send.bind(server.ws);
  server.ws.send=(message,...args)=>{if(message?.type==='full-reload')reloaded=true;return send(message,...args);};
  // Allow the filesystem watcher's initial scan to finish before editing a helper.
  await new Promise(done=>setTimeout(done,100));
  await fs.writeFile(helper,"export const syntaxCheck='changed version';\n");
  for(let i=0;!reloaded&&i<100;i++)await new Promise(done=>setTimeout(done,50));
  assert(reloaded,'A changed runtime helper must reload development clients and drop stale evidence');
  assert.notEqual(await fs.readFile(generated,'utf8'),before,'The executing worker identity must change with its dependency');
  console.log('PASS development worker updates: real helper edit regenerates immutable bytes/identity and reloads clients');
} finally {
  await server?.close();await fs.rm(fixture,{recursive:true,force:true});
}
