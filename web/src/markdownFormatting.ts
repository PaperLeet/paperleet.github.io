export type MarkdownFormat = 'bold' | 'italic' | 'code' | 'codeblock' | 'list' | 'link' | 'math';
export type MarkdownEdit = {before:string;after:string;start:number;end:number};

/** UTF-16 offsets match textarea selection; never split an existing emoji. */
function boundary(text:string,offset:number,direction:-1|1) {
  const position=Math.max(0,Math.min(text.length,Math.trunc(offset)));
  return position>0&&position<text.length&&/[\uD800-\uDBFF]/.test(text[position-1])&&/[\uDC00-\uDFFF]/.test(text[position])
    ?position+direction:position;
}
export function formatMarkdown(text:string,format:MarkdownFormat,start:number,end:number,limit:number,english=false):MarkdownEdit|undefined {
  if(!Number.isFinite(start)||!Number.isFinite(end)||!Number.isFinite(limit)||limit<0)return;
  let from=boundary(text,Math.min(start,end),-1),to=boundary(text,Math.max(start,end),1);
  const selected=text.slice(from,to);
  let replacement='',selectionStart=0,selectionEnd=0;
  const wrap=(prefix:string,suffix:string,placeholder:string)=>{
    const content=selected||placeholder;
    replacement=prefix+content+suffix;selectionStart=prefix.length;selectionEnd=prefix.length+content.length;
  };
  switch(format){
    case 'bold':wrap('**','**',english?'bold text':'加粗文字');break;
    case 'italic':wrap('*','*',english?'italic text':'斜体文字');break;
    case 'code':{
      const content=selected||(english?'code':'代码');
      const fence='`'.repeat(Math.max(1,...(content.match(/`+/g)||[]).map(value=>value.length+1)));
      const pad=content.startsWith('`')||content.endsWith('`')||(/^ .+ $/s.test(content)&&content.trim())?' ':'';
      wrap(fence+pad,pad+fence,english?'code':'代码');break;
    }
    case 'link':{
      const label=(selected||(english?'link text':'链接文字')).replace(/[\\[\]]/g,'\\$&');
      const prefix=`[${label}](`;
      replacement=prefix+'https://)';selectionStart=prefix.length;selectionEnd=prefix.length+8;break;
    }
    case 'list':{
      from=from===0?0:text.lastIndexOf('\n',from-1)+1;
      // Selecting through a newline must not include the following untouched line.
      const last=to>from&&text[to-1]==='\n'?to-1:to;
      const newline=text.indexOf('\n',last);
      to=newline===-1?text.length:newline;
      const content=text.slice(from,to)||(english?'list item':'列表内容');
      replacement=content.replace(/(^|\n)/g,'$1- ');selectionStart=0;selectionEnd=replacement.length;break;
    }
    case 'codeblock':
    case 'math':{
      const content=selected||(format==='math'?'\\alpha / r':english?'your code':'在这里写代码');
      const fence=format==='math'?'$$':'`'.repeat(Math.max(3,...(content.match(/`+/g)||[]).map(value=>value.length+1)));
      const before=from&&text[from-1]!=='\n'?'\n\n':from>1&&text[from-2]!=='\n'?'\n':'';
      const after=to<text.length&&text[to]!=='\n'?'\n\n':to+1<text.length&&text[to+1]!=='\n'?'\n':'';
      const prefix=before+fence+(format==='codeblock'?'python':'')+'\n';
      replacement=prefix+content+(content.endsWith('\n')?'':'\n')+fence+after;
      selectionStart=prefix.length;selectionEnd=prefix.length+content.length;break;
    }
    default:return;
  }
  const after=text.slice(0,from)+replacement+text.slice(to);
  if(after.length>limit)return;
  return {before:text,after,start:from+selectionStart,end:from+selectionEnd};
}
