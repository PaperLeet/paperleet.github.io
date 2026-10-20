/** Reordering is always a permutation of the complete captured list, including retired items. */
export const sameStudyListOrder=(a:readonly string[],b:readonly string[])=>a.length===b.length&&a.every((id,index)=>id===b[index]);
export function moveStudyListItem(ids:readonly string[],id:string,to:number){
 const from=ids.indexOf(id);
 if(from<0||!Number.isSafeInteger(to)||to<0||to>=ids.length||from===to||new Set(ids).size!==ids.length)return;
 const next=[...ids];next.splice(from,1);next.splice(to,0,id);return next;
}
export function canSaveStudyListOrder(source:readonly string[],draft:readonly string[],current:readonly string[]){
 return sameStudyListOrder(source,current)&&draft.length===source.length&&new Set(draft).size===draft.length&&draft.every(id=>source.includes(id))&&!sameStudyListOrder(source,draft);
}
