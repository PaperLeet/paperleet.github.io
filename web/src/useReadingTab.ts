import {useState} from 'react';

/** Retire a previous account's selection before its newly scoped children can commit. */
export function useReadingTab(scope:string){
 const [state,save]=useState({scope,tab:'description'});
 if(state.scope!==scope)save({scope,tab:'description'});
 return [state.scope===scope?state.tab:'description',(tab:string)=>save(previous=>({...previous,tab}))] as const;
}
