/** Narrow workspaces let the page scroll; desktop workspaces own a scroll pane. */
export function readingScrollRoot(node:HTMLElement):HTMLElement|null {
  const pane=node.closest<HTMLElement>('.description-scroll');
  if(!pane)return null;
  const style=window.getComputedStyle(pane),overflow=style.overflowY||style.overflow;
  return overflow==='visible'||overflow==='clip'?null:pane;
}

/** Only the current reader's toolbar can reserve space in its own scroll pane. */
export function readingToolbar(node:HTMLElement):HTMLElement|null {
  return node.closest('.solution-reading-content')?.querySelector<HTMLElement>('.solution-reading-toolbar')||null;
}

export function readingToolbarInset(node:HTMLElement):number {
  const pane=readingScrollRoot(node),toolbar=readingToolbar(node);
  if(!toolbar)return 24;
  const top=Number.parseFloat(window.getComputedStyle(toolbar).top)||0;
  return Math.max(24,Math.min(top+Math.ceil(toolbar.getBoundingClientRect().height)+12,(pane?pane.clientHeight:window.innerHeight)-44));
}

export function revealReadingSection(target:HTMLElement) {
  const pane=readingScrollRoot(target);
  if(!pane&&!readingToolbar(target)){target.scrollIntoView({block:'start'});return;}
  revealReadingTarget(target);
}

export function revealReadingTarget(target:HTMLElement,block:ScrollLogicalPosition='start') {
  const pane=readingScrollRoot(target);
  if(!pane){
    if(block==='start'&&readingToolbar(target))window.scrollTo({top:Math.max(0,window.scrollY+target.getBoundingClientRect().top-readingToolbarInset(target)),behavior:'instant'});
    else target.scrollIntoView({block,behavior:'instant'});
    return;
  }
  if(block==='nearest'){keepReadingTriggerVisible(target);return;}
  const bounds=target.getBoundingClientRect(),top=bounds.top-pane.getBoundingClientRect().top-pane.clientTop;
  const offset=block==='center'&&bounds.height<=pane.clientHeight?(pane.clientHeight-bounds.height)/2
    :block==='end'?pane.clientHeight-bounds.height-24:readingToolbarInset(target);
  pane.scrollTo({top:Math.max(0,pane.scrollTop+top-offset),behavior:'instant'});
}

export type ReadingAnchor={viewport:HTMLElement|null;top:number};
export function captureReadingAnchor(target:HTMLElement):ReadingAnchor {
  const viewport=readingScrollRoot(target);
  return {viewport,top:target.getBoundingClientRect().top-(viewport?viewport.getBoundingClientRect().top+viewport.clientTop:0)};
}

/** Restore in the original surface; a moved/removed pane must not scroll another view. */
export function restoreReadingAnchor(target:HTMLElement,origin:ReadingAnchor):boolean {
  if(!target.isConnected)return false;
  const current=captureReadingAnchor(target),pane=current.viewport;
  if(pane!==origin.viewport||pane&&!pane.isConnected)return false;
  const top=pane?Math.min(origin.top,Math.max(0,pane.clientHeight-44)):origin.top;
  if(pane)pane.scrollTo({top:Math.max(0,pane.scrollTop+current.top-top),behavior:'instant'});
  else window.scrollTo({top:Math.max(0,window.scrollY+current.top-top),behavior:'instant'});
  return true;
}

export function keepReadingTriggerVisible(target:HTMLElement,pageInset=0) {
  const pane=readingScrollRoot(target),bounds=target.getBoundingClientRect();
  const top=pane?pane.getBoundingClientRect().top+pane.clientTop:pageInset;
  const bottom=pane?top+pane.clientHeight:window.innerHeight;
  if(bounds.top>=top&&bounds.bottom<=bottom)return;
  if(!pane){target.scrollIntoView({block:'nearest',behavior:'instant'});return;}
  const delta=bounds.top<top?bounds.top-top:bounds.bottom-bottom;
  pane.scrollTo({top:Math.max(0,pane.scrollTop+delta),behavior:'instant'});
}
