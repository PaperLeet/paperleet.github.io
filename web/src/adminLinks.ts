import {adminOptions,readAdminRoute,type AdminView} from './adminNavigation';
export function adminHref(view:AdminView){
 if(!Number.isSafeInteger(view.page)||view.page<1||view.page>501)throw Error('Invalid administrator page');
 const params=new URLSearchParams(),options=adminOptions[view.section]||{};
 if(view.q)params.set('q',view.q);
 if(view.board)params.set('board',view.board);
 for(const key of ['status','review','kind','state','minimum'] as const)if(view[key]&&view[key]!==options[key]?.[0])params.set(key,view[key]!);
 if(view.page>1)params.set('page',String(view.page));
 if(view.item)params.set('item',view.item);
 if(view.create)params.set('create','1');
 const href='#/community/admin'+(view.section?'/'+view.section:'')+(params.size?'?'+params:'');
 if(!readAdminRoute(href))throw Error('Invalid administrator view');return href;
}
export const adminListView=(view:AdminView):AdminView=>({...view,item:undefined,create:undefined});

/** Navigate only to a known workbench object, never to an audit-supplied URL. */
export function adminAuditHref(kind:string,resource:string,detailsAvailable:boolean):string|undefined {
 if(kind==='highlights'){const match=resource.match(/^(?:global|board:([a-z][a-z0-9_-]{0,39})):[1-3]$/);return match?adminHref({section:'highlights',page:1,...(match[1]?{board:match[1]}:{})}):undefined;}
 if(!['boards','tags','users','papers'].includes(kind))return;
 const taxonomy=kind==='boards'||kind==='tags';
 if(!(taxonomy?/^[a-z][a-z0-9_-]{0,39}$/:/^[a-f0-9]{32}$/).test(resource))return;
 return adminHref({section:kind as 'boards'|'tags'|'users'|'papers',page:1,
  ...(kind==='papers'?{status:'all'}:{}),...(taxonomy||detailsAvailable?{item:resource}:{})});
}
