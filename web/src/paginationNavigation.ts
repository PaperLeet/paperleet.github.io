import type {KeyboardEvent} from 'react';

/** Keep the page window bounded even for very large result sets. */
export function paginationWindow(page:number,pages:number):(number|'gap')[]{
 if(!Number.isSafeInteger(pages)||pages<1||!Number.isSafeInteger(page))return [];
 const current=Math.max(1,Math.min(page,pages)),values=new Set([1,pages]);
 const start=current<=4?2:current>=pages-3?Math.max(2,pages-4):current-1;
 const end=current<=4?Math.min(pages-1,5):current>=pages-3?pages-1:current+1;
 for(let n=start;n<=end;n++)values.add(n);
 const result:(number|'gap')[]=[];
 for(const n of [...values].sort((a,b)=>a-b)){
  const previous=result.at(-1);
  if(typeof previous==='number'&&n-previous>1)result.push(n-previous===2?previous+1:'gap');
  result.push(n);
 }
 return result;
}

export function isPaginationActive(root:HTMLElement|null){
 return !!root?.isConnected&&!root.closest('[hidden],[inert],[aria-hidden="true"]');
}

/** Move focus, leaving page activation to the native button. */
export function focusPaginationButton(event:KeyboardEvent<HTMLDivElement>){
 if(event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
 if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
 const strip=event.currentTarget,target=event.target,view=strip.ownerDocument.defaultView;
 if(!view||!(target instanceof view.HTMLButtonElement)||target!==strip.ownerDocument.activeElement||!target.hasAttribute('data-pagination-control')||!strip.isConnected||strip.closest('[hidden],[inert],[aria-hidden="true"]'))return;
 const buttons=[...strip.querySelectorAll<HTMLButtonElement>('button[data-pagination-control]')].filter(button=>!button.disabled&&view.getComputedStyle(button).display!=='none');
 const index=buttons.indexOf(target);if(index<0)return;
 const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:Math.max(0,Math.min(buttons.length-1,index+(event.key==='ArrowRight'?1:-1)));
 event.preventDefault();buttons[next]?.focus({preventScroll:true});
}
