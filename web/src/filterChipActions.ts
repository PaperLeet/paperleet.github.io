export type FilterChipActions={canAct:()=>boolean;focusFallback:()=>HTMLElement|null};

const available=(node:HTMLElement)=>node.isConnected&&!node.closest('[hidden],[aria-hidden="true"],[inert]')&&!node.matches(':disabled');

/** Move focus before removing its native button; never schedule a later focus steal. */
export function applyFilterChip(button:HTMLButtonElement,{canAct,focusFallback}:FilterChipActions,change:()=>void){
 if(!available(button)||!canAct())return;
 if(document.activeElement===button){
  const buttons=[...button.parentElement!.querySelectorAll<HTMLButtonElement>('button[data-filter-key]')],index=buttons.indexOf(button);
  const target=index>=0?buttons.slice(index+1).find(available)||buttons.slice(0,index).reverse().find(available)||focusFallback():focusFallback();
  if(target&&available(target))target.focus({preventScroll:true});
 }
 change();
}
