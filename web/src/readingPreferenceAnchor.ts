import {captureReadingAnchor,readingScrollRoot,readingToolbarInset,type ReadingAnchor} from './readingViewport';

export type PreferenceAnchor={node:HTMLElement;origin:ReadingAnchor};
/** Pick a visible text block inside this reader, never an editor or another article. */
export function readingPreferenceAnchor(root:HTMLElement):PreferenceAnchor|undefined{
 const body=root.matches('.prose')?root:root.querySelector<HTMLElement>('.prose');
 if(!body)return;
 const pane=readingScrollRoot(body),edge=pane?pane.getBoundingClientRect().top+pane.clientTop:0;
 let top=edge+readingToolbarInset(body),bottom=pane?Math.min(window.innerHeight,edge+pane.clientHeight):window.innerHeight;
 const visual=window.visualViewport;if(visual&&visual.height>0){top=Math.max(top,visual.offsetTop);bottom=Math.min(bottom,visual.offsetTop+visual.height);}
 const header=body.closest('.app')?.querySelector<HTMLElement>(':scope > .topbar');
 if(header){const position=window.getComputedStyle(header).position,box=header.getBoundingClientRect();if((position==='sticky'||position==='fixed')&&box.top<=top)top=Math.max(top,box.bottom);}
 if(bottom<=top)return;
 const node=Array.from(body.querySelectorAll<HTMLElement>(':scope > p,:scope > h1,:scope > h2,:scope > h3,:scope > ul,:scope > ol,:scope > blockquote,:scope > table')).find(node=>{const box=node.getBoundingClientRect();return box.height>0&&box.bottom>top+8&&box.top<bottom;});
 return node?{node,origin:captureReadingAnchor(node)}:undefined;
}
