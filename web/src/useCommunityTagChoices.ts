import {useLayoutEffect,useRef,useState} from 'react';
import {matchingCommunityTags,communityTagQuery,COMMUNITY_TAG_PAGE_SIZE} from './communityTagBrowsing';
import type {CommunityTerm} from './communityTaxonomy';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {keepReadingTriggerVisible} from './readingViewport';

const normalized=(value:string)=>communityTagQuery(value).normalize('NFKC').toLowerCase().trim().replace(/\s+/g,' ');

/** Local metadata browsing never changes a selection, feed scope or publication. */
export function useCommunityTagChoices(tags:readonly CommunityTerm[],selected?:string|readonly string[],enabled=true){
 const [state,setState]=useState({query:'',limit:COMMUNITY_TAG_PAGE_SIZE});
 const rows=matchingCommunityTags(tags,state.query,selected),visible=rows.slice(0,state.limit),more=Math.min(COMMUNITY_TAG_PAGE_SIZE,rows.length-visible.length),expanded=visible.length>COMMUNITY_TAG_PAGE_SIZE;
 const root=useRef<HTMLDivElement>(null),compactControl=useRef<HTMLButtonElement>(null);
 const scope=JSON.stringify([enabled,normalized(state.query),rows.map(row=>[row.id,row.enabled])]),intent=useWorkspaceIntent(scope);
 const pending=useRef<{id?:string;claim:()=>boolean}|undefined>(undefined);
 function setQuery(query:string){if(!enabled)return;pending.current=undefined;intent.cancel();setState(current=>({query,limit:normalized(query)===normalized(current.query)?current.limit:COMMUNITY_TAG_PAGE_SIZE}));}
 function expand(){if(!enabled||!more)return;pending.current={id:rows[visible.length].id,claim:intent.create()};setState(current=>({...current,limit:visible.length+more}));}
 function compact(){if(!enabled||!expanded)return;pending.current={claim:intent.create()};setState(current=>({...current,limit:COMMUNITY_TAG_PAGE_SIZE}));}
 useLayoutEffect(()=>{
  const request=pending.current;if(!request)return;pending.current=undefined;
  const row=request.id?Array.from(root.current?.querySelectorAll<HTMLElement>('[data-tag-option]')||[]).find(node=>node.dataset.tagOption===request.id):undefined;
  const target=row?(row.matches('button')?row:row.querySelector<HTMLElement>('input:not(:disabled)')||row):request.id?null:compactControl.current;
  if(target?.isConnected&&!target.closest('[hidden],details:not([open])')&&request.claim()){target.focus({preventScroll:true});keepReadingTriggerVisible(target,100);}else intent.cancel();
 },[state.limit,scope]);
 return {query:state.query,setQuery,rows,visible,more,expanded,root,compactControl,expand,compact};
}
