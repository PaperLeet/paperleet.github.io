import {createContext,useContext,useMemo} from 'react';
import type {ReactNode} from 'react';
import {useAccount} from './social';
import {status,type Problem,type Saved} from './types';

type ModuleProgress={state:ReturnType<typeof status>;updated:string};
type PracticeProgress={owner:string;ready:boolean;failed:boolean;read:(id:string)=>ModuleProgress|undefined;retry?:()=>void};
const Context=createContext<PracticeProgress|undefined>(undefined);

/** Read the already-loaded workspace; readers never start a second sync or receive source code. */
export function PracticeProgressProvider({owner,problems,saved,ready,failed,retry,children}:{owner:string;problems:Problem[];saved:Record<string,Saved>;ready:boolean;failed:boolean;retry:()=>void;children:ReactNode}){
 const {user,health,loading}=useAccount();
 const expected=loading?'unresolved':health?.mode==='local'?'local':health?.mode==='hosted'?user?'user:'+user.id:'anonymous':'unresolved';
 const matched=owner!=='unresolved'&&owner===expected;
 const byId=useMemo(()=>new Map(problems.map(problem=>[problem.id,problem])),[problems]);
 const value=useMemo<PracticeProgress>(()=>({
  owner:matched?owner:'unresolved',ready:matched&&ready&&!failed,failed:matched&&failed,retry:matched?retry:undefined,
  read(id){
   if(!matched||!ready||failed)return;
   const problem=byId.get(id);if(!problem)return;
   const record=saved[id];
   return {state:status(problem,record),updated:typeof record?.updated==='string'?record.updated:''};
  },
 }),[owner,matched,ready,failed,byId,saved,retry]);
 return <Context.Provider value={value}>{children}</Context.Provider>;
}
export const usePracticeProgress=()=>useContext(Context);
