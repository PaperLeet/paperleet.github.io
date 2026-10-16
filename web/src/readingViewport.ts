/** The workspace reads inside its left pane; standalone discussions scroll the page. */
export function readingScrollRoot(node:HTMLElement):HTMLElement|null {
  return node.closest<HTMLElement>('.description-scroll');
}

export function revealReadingSection(target:HTMLElement) {
  const pane=readingScrollRoot(target);
  if(!pane){target.scrollIntoView({block:'start'});return;}
  const top=target.getBoundingClientRect().top-pane.getBoundingClientRect().top-pane.clientTop;
  pane.scrollTo({top:Math.max(0,pane.scrollTop+top-24),behavior:'instant'});
}

export function keepReadingTriggerVisible(target:HTMLElement) {
  const pane=readingScrollRoot(target),bounds=target.getBoundingClientRect();
  const top=pane?pane.getBoundingClientRect().top+pane.clientTop:0;
  const bottom=pane?top+pane.clientHeight:window.innerHeight;
  if(bounds.top>=top&&bounds.bottom<=bottom)return;
  if(!pane){target.scrollIntoView({block:'nearest',behavior:'instant'});return;}
  const delta=bounds.top<top?bounds.top-top:bounds.bottom-bottom;
  pane.scrollTo({top:Math.max(0,pane.scrollTop+delta),behavior:'instant'});
}
