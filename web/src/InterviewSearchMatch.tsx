import {Fragment,memo} from 'react';
import {interviewSearchTerms} from './interviewSearch';
import {publicSourceSearchRanges} from './replySearchPassages';
import './interview-search-match.css';

/** Preserve the exact public source text while marking literal, normalized words. */
export default memo(function InterviewSearchMatch({text,query}:{text:string;query:string}){
 const ranges=publicSourceSearchRanges(text,interviewSearchTerms(query));let cursor=0;
 return <>{ranges.map(({start,end})=>{const before=text.slice(cursor,start);cursor=end;return <Fragment key={start}>{before}<mark className="interview-concept-mark">{text.slice(start,end)}</mark></Fragment>;})}{text.slice(cursor)}</>;
});
