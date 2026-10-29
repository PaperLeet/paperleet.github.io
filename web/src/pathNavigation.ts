import {appLocation, publicHref, routeHash, installPathRouter, type CatalogRoute} from './appLocation';
import {applySearchMetadata, searchMetadata, type SearchCatalog} from './searchMetadata';

/** History routing is installed before React; legacy bookmarks remain usable. */
export function startPathNavigation() {
  const modules=new Map<string,string>(),numbers=new Map<string,string>();
  const directoryPath=(pathname:string)=>pathname.replace(/\/index\.html$/, '/');
  const knownPath=(pathname:string)=>/^\/(?:$|(?:papers|collections)\/[a-z0-9_-]+\/?$|problems\/[a-z0-9_-]+\/[a-z0-9_-]+\/?$)/.test(directoryPath(pathname));
  function read(value:string|URL) {
    const url=new URL(value,location.href);
    if(url.hash.startsWith('#/'))return url.hash;
    // Pages serves index.html aliases with HTTP 200. Read them as their directory
    // so startup preserves the page's self-canonical metadata and app route.
    let path=directoryPath(url.pathname).replace(/^\/+|\/+$/g,'');
    if(path.startsWith('problems/')){path=path.slice(9);path=numbers.get(path)||path;}
    const params=new URLSearchParams(url.search);
    for(const key of [...params.keys()])if(key==='_paperleet_reload'||key.startsWith('utm_')||key==='gclid'||key==='fbclid')params.delete(key);
    return (path?'#/'+path:'#')+(params.size?(path?'?':'/?')+params:'');
  }
  function href(value:string|undefined) {
    if(value===undefined||value!=='#'&&!value.startsWith('#/'))return value;
    const raw=value.replace(/^#\/?/,'').replace(/^lists\//,'collections/'),split=raw.indexOf('?');
    const path=split<0?raw:raw.slice(0,split),query=split<0?'':raw.slice(split);
    if(!path)return '/'+query;
    if(/^(papers|collections)\/[a-z0-9_-]+$/.test(path))return '/'+path+'/'+query;
    const module=modules.get(path)||(/^(foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval|self_evolving|network_biology|multimodal|robotics|ai_infra|diffusion_rl)\/[a-z0-9_-]+$/.test(path)?path:undefined);
    return module?'/problems/'+module+'/'+query:'/'+value;
  }
  function navigate(value:string) {
    const target=href(value.startsWith('#')?value:'#'+value)!;
    const oldURL=location.href;if(new URL(target,oldURL).href===oldURL)return;
    history.pushState(history.state,'',target);
    window.dispatchEvent(new HashChangeEvent('hashchange',{oldURL,newURL:location.href}));
  }
  function register(problems:CatalogRoute[]) {
    for(const problem of problems){modules.set(String(problem.number),problem.id);modules.set(problem.id,problem.id);numbers.set(problem.id,String(problem.number));}
    if(location.hash&&!location.hash.startsWith('#/'))return;
    const target=href(read(location.href))!;
    if(new URL(target,location.href).href!==location.href)history.replaceState(history.state,'',target);
  }
  installPathRouter({read,href,navigate,register,sameUrl:url=>url.origin===location.origin&&knownPath(url.pathname)});

  let catalog: SearchCatalog | undefined;
  const updateMetadata=()=>{if(catalog)applySearchMetadata(searchMetadata(appLocation.hash,catalog));};
  document.addEventListener('paperleet:catalog', event=>{catalog=(event as CustomEvent<SearchCatalog>).detail;updateMetadata();});
  const normalize = () => {
    if (location.hash && !location.hash.startsWith('#/')) return;
    const href = publicHref(appLocation.hash)!;
    if (new URL(href, location.href).href !== location.href) history.replaceState(history.state, '', href);
  };
  normalize();
  let previous = location.href;
  window.addEventListener('hashchange', event => {
    // Back/forward can emit both popstate and hashchange for the same traversal.
    if (event.isTrusted && location.href === previous) { event.stopImmediatePropagation(); return; }
    if (read(event.oldURL) === read(event.newURL)) { previous = location.href; event.stopImmediatePropagation(); return; }
    normalize();
    previous = location.href;
    updateMetadata();
  }, true);
  window.addEventListener('popstate', () => {
    const oldURL = previous;
    normalize();
    if (read(oldURL) === read(location.href)) { previous = location.href; return; }
    if (oldURL !== location.href) window.dispatchEvent(new HashChangeEvent('hashchange', {oldURL, newURL: location.href}));
  });
  window.addEventListener('click', event => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || !(event.target instanceof Element)) return;
    const link = event.target.closest<HTMLAnchorElement>('a[href]');
    if (!link || link.hasAttribute('download') || link.target && link.target !== '_self') return;
    const target = new URL(link.href, location.href);
    if (target.origin !== location.origin || !knownPath(target.pathname)) return;
    // Plain document anchors are not app routes.
    if (target.hash && !/^#\//.test(target.hash) && target.hash !== '#') return;
    event.preventDefault();
    appLocation.hash = routeHash(target);
  });
}
