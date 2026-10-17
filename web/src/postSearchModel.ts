/** Match the community's literal ASCII-insensitive search, without user-built regular expressions. */
const fold=(value:string)=>value.replace(/[A-Z]/g,letter=>letter.toLowerCase());
export function postSearchQuery(value?:string){
 if(typeof value!=='string')return '';
 const query=value.trim(),characters=Array.from(query);
 return characters.length<=200&&!characters.some(letter=>{const point=letter.codePointAt(0)!;return point>=0xd800&&point<=0xdfff;})?query:'';
}
export function postSearchIndex(text:string,query:string){
 const needle=postSearchQuery(query);
 return needle?fold(text).indexOf(fold(needle)):-1;
}
let segmenter:Intl.Segmenter|undefined;
function boundaries(text:string){
 if(typeof Intl.Segmenter==='function'){
  segmenter||=new Intl.Segmenter(undefined,{granularity:'grapheme'});
  return Array.from(segmenter.segment(text),row=>({start:row.index,end:row.index+row.segment.length}));
 }
 let offset=0;
 return Array.from(text,letter=>{const start=offset;offset+=letter.length;return {start,end:offset};});
}

export function postSearchWindow(text:string,query:string,limit:number){
 const index=postSearchIndex(text,query);if(index<0)return;
 const letters=boundaries(text),first=letters.findIndex(row=>row.end>index),last=letters.findIndex(row=>row.end>=index+query.trim().length);
 if(first<0||last<first)return;
 const requested=Number.isSafeInteger(limit)?Math.max(1,Math.min(400,limit)):210;
 const width=Math.max(requested,last-first+1);
 const start=letters.length<=width?0:Math.max(0,first-Math.min(24,Math.floor((width-(last-first+1))/3))),end=Math.min(letters.length,start+width);
 const excerpt=(start?'…':'')+text.slice(letters[start].start,letters[end-1].end)+(end<letters.length?'…':'');
 return {text:excerpt,truncated:start>0||end<letters.length,searchRange:{start:start?1:0,end:excerpt.length-(end<letters.length?1:0)}};
}

/** Whole graphemes stay intact, including partial matches inside a combined emoji/accent. */
export function postSearchRanges(text:string,query:string){
 const needle=postSearchQuery(query);if(!needle)return [];
 const haystack=fold(text),wanted=fold(needle),letters=boundaries(text),ranges:{start:number;end:number}[]=[];
 let cursor=0;
 for(let count=0;count<16;count++){
  const index=haystack.indexOf(wanted,cursor);if(index<0)break;
  const start=letters.find(row=>row.end>index)?.start,end=letters.find(row=>row.end>=index+wanted.length)?.end;
  if(start===undefined||end===undefined)break;
  const previous=ranges.at(-1);
  if(previous&&start<=previous.end)previous.end=Math.max(previous.end,end);else ranges.push({start,end});
  cursor=index+wanted.length;
 }
 return ranges;
}
