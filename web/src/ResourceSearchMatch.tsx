import {Fragment} from 'react';
import {publicSourceSearchRanges} from './replySearchPassages';
import {resourceBrowserTerms} from './interviewResourceBrowsing';

export default function ResourceSearchMatch({text,query}:{text:string;query:string}){
 const ranges=publicSourceSearchRanges(text,resourceBrowserTerms(query).split(' ').filter(Boolean));let cursor=0;
 return <>{ranges.map(({start,end})=>{const before=text.slice(cursor,start);cursor=end;return <Fragment key={start}>{before}<mark className="resource-search-match">{text.slice(start,end)}</mark></Fragment>;})}{text.slice(cursor)}</>;
}
