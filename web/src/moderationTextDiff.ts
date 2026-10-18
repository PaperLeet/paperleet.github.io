export type DiffLine={kind:'context'|'removed'|'added';before?:number;after?:number;text:string};
export type TextDiff={mode:'lines'|'block';lines:DiffLine[];added:number;removed:number};
export type DiffDisplayRow=DiffLine|{kind:'gap';count:number};
const maxCells=250_000;
export const MAX_DIFF_SOURCE=80_000,MAX_DIFF_ROWS=400;
const sourceLines=(value:string)=>value.match(/[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$/g)||[];

/** Preserve every source character, including blank lines and line endings.
 * A bounded LCS aligns small changes; large middle blocks use an honest full
 * replacement, never an expensive or approximate claim of minimal edits. */
export function moderationTextDiff(before:string,after:string):TextDiff|undefined{
 if(before.length>MAX_DIFF_SOURCE||after.length>MAX_DIFF_SOURCE)return;
 const old=sourceLines(before),next=sourceLines(after),lines:DiffLine[]=[];
 let prefix=0,suffix=0;
 while(prefix<old.length&&prefix<next.length&&old[prefix]===next[prefix])prefix++;
 while(suffix<old.length-prefix&&suffix<next.length-prefix&&old[old.length-1-suffix]===next[next.length-1-suffix])suffix++;
 let i=0,j=0,added=0,removed=0;
 const emit=(kind:DiffLine['kind'])=>{
  if(kind==='context'){lines.push({kind,before:i+1,after:j+1,text:old[i]});i++;j++;}
  else if(kind==='removed'){lines.push({kind,before:i+1,text:old[i++]});removed++;}
  else{lines.push({kind,after:j+1,text:next[j++]});added++;}
 };
 while(i<prefix)emit('context');
 const m=old.length-prefix-suffix,n=next.length-prefix-suffix,mode=(m+1)*(n+1)<=maxCells?'lines':'block';
 if(mode==='lines'){
  const width=n+1,scores=new Uint16Array((m+1)*width);
  for(let a=m-1;a>=0;a--)for(let c=n-1;c>=0;c--)scores[a*width+c]=old[prefix+a]===next[prefix+c]?1+scores[(a+1)*width+c+1]:Math.max(scores[(a+1)*width+c],scores[a*width+c+1]);
  while(i<prefix+m||j<prefix+n){
   if(i<prefix+m&&j<prefix+n&&old[i]===next[j])emit('context');
   else if(i<prefix+m&&(j===prefix+n||scores[(i-prefix+1)*width+j-prefix]>=scores[(i-prefix)*width+j-prefix+1]))emit('removed');
   else emit('added');
  }
 }else{
  while(i<old.length-suffix)emit('removed');
  while(j<next.length-suffix)emit('added');
 }
 while(i<old.length)emit('context');
 return {mode,lines,added,removed};
}

/** Changed rows plus three context lines. Rendering is bounded separately
 * from comparison; both complete raw sources remain available to the reader. */
export function moderationDiffRows(diff:TextDiff){
 const ranges:{start:number;end:number}[]=[];
 diff.lines.forEach((line,index)=>{
  if(line.kind==='context')return;
  const start=Math.max(0,index-3),end=Math.min(diff.lines.length,index+4),last=ranges.at(-1);
  if(last&&start<=last.end)last.end=Math.max(last.end,end);else ranges.push({start,end});
 });
 const rows:DiffDisplayRow[]=[];let cursor=0,shown=0;
 for(const range of ranges){
  if(range.start>cursor)rows.push({kind:'gap',count:range.start-cursor});
  for(let index=range.start;index<range.end;index++){
   if(shown===MAX_DIFF_ROWS)return {rows,truncated:true};
   rows.push(diff.lines[index]);shown++;
  }
  cursor=range.end;
 }
 if(ranges.length&&cursor<diff.lines.length)rows.push({kind:'gap',count:diff.lines.length-cursor});
 return {rows,truncated:false};
}

export function diffLineContent(value:string){
 const ending=/\r\n$/.test(value)?'CRLF':/\r$/.test(value)?'CR':/\n$/.test(value)?'LF':'EOF';
 return {text:value.replace(/\r\n$|[\r\n]$/,''),ending};
}
