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
  revealReadingTarget(target);
}

export function revealReadingTarget(target:HTMLElement,block:ScrollLogicalPosition='start') {
  if(!target.isConnected||target.closest('[hidden]'))return;
  const pane=readingScrollRoot(target);
  if(block==='nearest'){keepReadingTriggerVisible(target);return;}
  if(!pane){
    const range=readingVisibleRange(target,null,0),bounds=target.getBoundingClientRect();
    if(range.top>0||range.bottom<window.innerHeight){
      if(range.bottom<=range.top||!Number.isFinite(bounds.top))return;
      const available=range.bottom-range.top;
      const offset=block==='center'&&bounds.height<=available?range.top+(available-bounds.height)/2
        :block==='end'?range.bottom-bounds.height:range.top;
      window.scrollTo({top:Math.max(0,window.scrollY+bounds.top-offset),behavior:'instant'});
    }else target.scrollIntoView({block,behavior:'instant'});
    return;
  }
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

/** The page and a nested reader each keep their own scroll position. */
function readingVisibleRange(target:HTMLElement,pane:HTMLElement|null,pageInset:number){
  const visual=window.visualViewport,screenHeight=window.innerHeight;
  const hasVisual=visual&&Number.isFinite(visual.offsetTop)&&Number.isFinite(visual.height)&&visual.height>0;
  const visualTop=hasVisual?Math.max(0,visual.offsetTop):0;
  const screenBottom=hasVisual?Math.min(screenHeight,visualTop+visual.height):screenHeight;
  const paneTop=pane?pane.getBoundingClientRect().top+pane.clientTop:0;
  const baseBottom=pane?paneTop+pane.clientHeight:screenBottom;
  const style=window.getComputedStyle(pane||document.documentElement);
  const pixels=(value:string)=>Math.max(0,Number.parseFloat(value)||0);
  let top=Math.max(visualTop,paneTop+pixels(style.scrollPaddingTop),pane?0:Number.isFinite(pageInset)?pageInset:0);
  let bottom=Math.min(screenBottom,baseBottom-pixels(style.scrollPaddingBottom));
  // Page scroll margins already express header clearance; do not add it twice
  // or apply that global page offset inside the independent workspace pane.
  if(!pane){const targetStyle=window.getComputedStyle(target);top=Math.max(top,pixels(targetStyle.scrollMarginTop));bottom=Math.min(bottom,screenBottom-pixels(targetStyle.scrollMarginBottom));}
  const header=target.closest('.app')?.querySelector<HTMLElement>(':scope > .topbar');
  if(header&&!header.contains(target)){
    const position=window.getComputedStyle(header).position,bounds=header.getBoundingClientRect();
    if((position==='sticky'||position==='fixed')&&bounds.height>0&&bounds.top<=top&&bounds.bottom>top)top=bounds.bottom;
  }
  const toolbar=readingToolbar(target);
  if(toolbar&&!toolbar.contains(target))top=Math.max(top,paneTop+readingToolbarInset(target));
  return {top,bottom};
}

/** Reveal only the nearest covered edge; an already visible reading position stays put. */
export function keepReadingTriggerVisible(target:HTMLElement,pageInset=0) {
  if(!target.isConnected||target.closest('[hidden]'))return;
  const pane=readingScrollRoot(target),bounds=target.getBoundingClientRect();
  if(!Number.isFinite(bounds.top)||!Number.isFinite(bounds.bottom)||bounds.bottom<=bounds.top)return;
  const {top,bottom}=readingVisibleRange(target,pane,pageInset);
  if(!Number.isFinite(top)||!Number.isFinite(bottom)||bottom<=top)return;
  const start=bounds.top,end=bounds.bottom;
  if(start>=top&&end<=bottom||start<=top&&end>=bottom)return;
  const fits=end-start<=bottom-top;
  const delta=start<top?(fits?start-top:end-bottom):(fits?end-bottom:start-top);
  if(!delta)return;
  if(pane)pane.scrollTo({top:Math.max(0,pane.scrollTop+delta),behavior:'instant'});
  else window.scrollTo({top:Math.max(0,window.scrollY+delta),behavior:'instant'});
}
