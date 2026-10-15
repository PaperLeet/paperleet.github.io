import {useEffect} from 'react';
import {resumePracticeReadingOrigin} from './readingReturn';

export function usePracticeReadingOrigin(owner:string|undefined,problem:{id:string;number:number}|undefined,href:string|undefined,route?:string){
 const id=problem?.id,number=problem?.number;
 const expected=route??(typeof location==='undefined'?'':location.hash);
 useEffect(()=>resumePracticeReadingOrigin(owner,id&&number?{id,number}:undefined,href,expected),[owner,id,number,href,expected]);
}
