import {adminOptions,readAdminRoute,type AdminView} from './adminNavigation';
export function adminHref(view:AdminView){
 if(!Number.isSafeInteger(view.page)||view.page<1||view.page>501)throw Error('Invalid administrator page');
 const params=new URLSearchParams(),options=adminOptions[view.section]||{};
 if(view.q)params.set('q',view.q);
 for(const key of ['status','review','kind'] as const)if(view[key]&&view[key]!==options[key]?.[0])params.set(key,view[key]!);
 if(view.page>1)params.set('page',String(view.page));
 if(view.item)params.set('item',view.item);
 if(view.create)params.set('create','1');
 const href='#/community/admin'+(view.section?'/'+view.section:'')+(params.size?'?'+params:'');
 if(!readAdminRoute(href))throw Error('Invalid administrator view');return href;
}
export const adminListView=(view:AdminView):AdminView=>({...view,item:undefined,create:undefined});
