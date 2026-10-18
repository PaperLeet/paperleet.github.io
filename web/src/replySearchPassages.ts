import {foldReplySearch,replySearchQuery,replySearchTerms} from './replySearch';

type Span={start:number;end:number;from:number;to:number};
export type ReplySearchPassage={text:string;before:boolean;after:boolean;ranges:{start:number;end:number}[]};
let segmenter:Intl.Segmenter|undefined;
function mapSource(text:string){
 const segments=typeof Intl.Segmenter==='function'?(segmenter||=new Intl.Segmenter(undefined,{granularity:'grapheme'})).segment(text):undefined;
 const spans:Span[]=[];let offset=0,foldedOffset=0;
 for(const segment of segments||Array.from(text)){
  const value=typeof segment==='string'?segment:segment.segment,start=typeof segment==='string'?offset:segment.index;
  const end=start+value.length,from=foldedOffset;foldedOffset+=foldReplySearch(value).length;
  spans.push({start,end,from,to:foldedOffset});offset=end;
 }
 const folded=foldReplySearch(text);
 return folded.length===foldedOffset?{spans,folded}:undefined;
}
function containing(spans:Span[],offset:number){
 let low=0,high=spans.length;
 while(low<high){const middle=(low+high)>>>1;if(spans[middle].to<=offset)low=middle+1;else high=middle;}
 return Math.min(low,spans.length-1);
}
function highlights(mapped:{spans:Span[];folded:string},terms:string[],first:number,last:number){
 const from=mapped.spans[first].from,to=mapped.spans[last].to,offset=mapped.spans[first].start,endOffset=mapped.spans[last].end;
 const found:{start:number;end:number}[]=[];
 for(const term of terms){let cursor=Math.max(0,from-term.length+1);for(let count=0;count<16;count++){
  const index=mapped.folded.indexOf(term,cursor);if(index<0||index>=to)break;
  const start=Math.max(offset,mapped.spans[containing(mapped.spans,index)].start)-offset,end=Math.min(endOffset,mapped.spans[containing(mapped.spans,index+term.length-1)].end)-offset;
  found.push({start,end});cursor=index+term.length;
 }}
 const ranges:{start:number;end:number}[]=[];
 for(const value of found.sort((a,b)=>a.start-b.start||b.end-a.end)){
  const previous=ranges.at(-1);
  if(previous&&value.start<=previous.end)previous.end=Math.max(previous.end,value.end);else if(ranges.length<24)ranges.push({...value});else break;
 }
 return ranges;
}

/** Literal public-source excerpts. Clipping markers never become search evidence. */
export function replySearchPassages(text:string,query:string):ReplySearchPassage[]{
 const terms=[...new Set(replySearchTerms(replySearchQuery(query)))];
 if(!terms.length||!text||text.length>100_000)return [];
 const mapped=mapSource(text);if(!mapped?.spans.length)return [];
 const matches=terms.flatMap(term=>{const index=mapped.folded.indexOf(term);return index<0?[]:[{first:containing(mapped.spans,index),last:containing(mapped.spans,index+term.length-1)}];}).sort((a,b)=>a.first-b.first||b.last-a.last);
 const windows:{first:number;last:number}[]=[];
 for(const match of matches){
  const previous=windows.at(-1);if(previous&&match.first>=previous.first&&match.last<=previous.last)continue;
  const first=Math.max(0,match.first-24),last=Math.min(mapped.spans.length-1,first+Math.min(239,Math.max(159,match.last-first)));
  if(previous&&first<=previous.last&&last-previous.first<240){previous.last=Math.max(previous.last,last);continue;}
  if(windows.length===3)break;windows.push({first,last});
 }
 return windows.map(({first,last})=>{
  const start=mapped.spans[first].start,end=mapped.spans[last].end,source=text.slice(start,end);
  return {text:source,before:start>0,after:end<text.length,ranges:highlights(mapped,terms,first,last)};
 });
}
