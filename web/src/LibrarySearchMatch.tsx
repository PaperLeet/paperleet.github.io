import {Fragment} from 'react';
import {postSearchIndex,postSearchRanges} from './postSearchModel';

/** The library matches separate literal terms, even when they occur in different fields. */
export default function LibrarySearchMatch({text,query}:{text:string;query:string}){
 const terms=[...new Set(query.trim().split(/\s+/).filter(Boolean))];
 if(!terms.length)return <>{text}</>;
 const ranges=terms.flatMap(term=>postSearchIndex(text,term)<0?[]:postSearchRanges(text,term)).sort((a,b)=>a.start-b.start||b.end-a.end);
 const merged:{start:number;end:number}[]=[];
 for(const range of ranges){
  const previous=merged.at(-1);
  if(previous&&range.start<=previous.end)previous.end=Math.max(previous.end,range.end);
  else merged.push({...range});
 }
 let cursor=0;
 return <>{merged.map(({start,end})=>{const before=text.slice(cursor,start);cursor=end;return <Fragment key={start}>{before}<mark className="library-search-match">{text.slice(start,end)}</mark></Fragment>;})}{text.slice(cursor)}</>;
}
