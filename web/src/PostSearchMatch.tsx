import {Fragment} from 'react';
import {postSearchRanges} from './postSearchModel';
import './post-search-match.css';

export default function PostSearchMatch({text,query='',range}:{text:string;query?:string;range?:{start:number;end:number}}){
 const from=range?.start||0,to=range?.end??text.length;
 const ranges=postSearchRanges(text.slice(from,to),query).map(({start,end})=>({start:start+from,end:end+from}));if(!ranges.length)return <>{text}</>;
 let previous=0;
 return <>{ranges.map(({start,end})=>{const before=text.slice(previous,start);previous=end;return <Fragment key={start}>{before}<mark className="post-search-match">{text.slice(start,end)}</mark></Fragment>;})}{text.slice(previous)}</>;
}
