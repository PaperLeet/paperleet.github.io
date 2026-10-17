import {useMemo,useState} from 'react';
import {ArrowDownToLine} from 'lucide-react';
import {useI18n} from './i18n';
import {writingSections,type WritingSection} from './writingSections';

type Props={source:string;ready:boolean;locked:boolean;onJump:(source:string,section:WritingSection,opener:HTMLElement)=>void};
export default function MarkdownSectionNav({source,ready,locked,onJump}:Props){
 const {t}=useI18n(),sections=useMemo(()=>writingSections(source),[source]);
 const [selected,setSelected]=useState<number>();
 const section=sections.find(row=>row.offset===selected)||sections[0];
 if(!sections.length)return <p className="markdown-section-empty">{t('添加 Markdown 小标题后，可以在这里快速定位。','Add Markdown section headings to navigate your writing here.')}</p>;
 return <div className="markdown-section-controls">
  <label>{t('选择正文章节','Choose a writing section')}<select value={section.offset} disabled={locked||!ready} onChange={event=>setSelected(Number(event.target.value))}>
   {sections.map((row,index)=><option key={row.offset} value={row.offset}>{`${index+1}. ${'· '.repeat(row.depth-1)}${row.label}`}</option>)}
  </select></label>
  <button type="button" className="text-button" disabled={locked||!ready} onClick={event=>{if(!locked&&ready)onJump(source,section,event.currentTarget);}}><ArrowDownToLine size={15} aria-hidden="true"/>{t('定位到正文','Go to section')}</button>
  {!ready&&<small role="status">{t('目录更新中…','Updating sections…')}</small>}
 </div>;
}
