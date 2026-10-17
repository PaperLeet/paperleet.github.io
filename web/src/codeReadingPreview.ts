/** Logical source lines; CRLF is one break and a terminal newline adds no empty line. */
export function codeReadingPreview(code:string) {
  let lines=code?1:0,breaks=0,cut=code.length;
  for(const match of code.matchAll(/\r\n|\r|\n/g)) {
    if(++breaks===16)cut=match.index;
    if(match.index+match[0].length<code.length)lines++;
  }
  const collapsible=lines>24;
  return {lines,collapsible,preview:collapsible?code.slice(0,cut):code};
}
