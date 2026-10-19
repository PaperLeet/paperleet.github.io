import type {KeyboardEvent} from 'react';

const inactive='[hidden],[aria-hidden="true"],[inert]';
export const activeSourcePreview=(node:HTMLElement|null|undefined)=>!!node?.isConnected&&!node.closest(inactive);

/** Editable children own Escape; only the nearest active preview handles an unused key. */
export function sourcePreviewEscape(event:KeyboardEvent<HTMLElement>){
 if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||!activeSourcePreview(event.currentTarget))return false;
 const target=event.target as HTMLElement;
 return typeof target.closest==='function'&&!target.closest(`${inactive},input,textarea,select,[contenteditable]:not([contenteditable="false"])`);
}
