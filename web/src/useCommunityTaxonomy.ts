import {useEffect,useSyncExternalStore} from 'react';
import {socialApi,useAccount,errorMessage} from './social';
import {fallbackTaxonomy,readCommunityTaxonomy,type CommunityTaxonomy} from './communityTaxonomy';
type Snapshot={data:CommunityTaxonomy;ready:boolean;loading:boolean;error:string};
let state:Snapshot={data:fallbackTaxonomy,ready:false,loading:false,error:''},inflight:{controller:AbortController;promise:Promise<void>;version:number}|undefined,version=0,consumers=0;
const listeners=new Set<()=>void>(),snapshot=()=>state;
const subscribe=(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener);};};
const update=(change:Partial<Snapshot>)=>{state={...state,...change};for(const listener of listeners)listener();};
export function refreshCommunityTaxonomy(){
 if(inflight)return inflight.promise;
 const controller=new AbortController(),request=++version;update({loading:true,error:''});const timer=setTimeout(()=>controller.abort(),20_000);
 const promise=Promise.resolve().then(async()=>{try{const data=readCommunityTaxonomy(await socialApi<CommunityTaxonomy>('community/taxonomy',undefined,controller.signal));if(request===version)update({data,ready:true});}catch(e){if(request===version)update({error:errorMessage(e)});}finally{clearTimeout(timer);if(inflight?.version===request)inflight=undefined;if(request===version)update({loading:false});}});
 inflight={controller,promise,version:request};return promise;
}
function focus(){if(document.visibilityState==='visible')void refreshCommunityTaxonomy();}
export function useCommunityTaxonomy(){
 const {capabilities,loading}=useAccount(),supported=capabilities?.community_taxonomy===true&&!loading,data=useSyncExternalStore(subscribe,snapshot,snapshot);
 useEffect(()=>{if(!supported)return;consumers++;if(consumers===1)window.addEventListener('focus',focus);if(!state.ready)void refreshCommunityTaxonomy();return()=>{consumers--;if(!consumers){window.removeEventListener('focus',focus);inflight?.controller.abort();inflight=undefined;version++;if(state.loading)update({loading:false});}};},[supported]);
 return supported?{...data,supported:true,refresh:refreshCommunityTaxonomy}:{data:fallbackTaxonomy,ready:true,loading:false,error:'',supported:false,refresh:refreshCommunityTaxonomy};
}
