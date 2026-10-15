import {useId,useState} from 'react';
import {Building2,ChevronDown,ExternalLink} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewLabel} from './interviewModels';
import {companyResources,focusInterviewSection} from './interviewPreparation';

export default function InterviewResources({company=''}:{company?:string}){
 const {t,locale}=useI18n(),en=locale==='en';
 const [expanded,setExpanded]=useState(false),extraId=useId();
 const resources=companyResources(company),visible=company?resources:resources.slice(0,3);
 const cards=(rows:typeof resources)=>rows.map(resource=><a key={resource.id} href={resource.url} target="_blank" rel="noreferrer"><span><Building2 size={14}/>{interviewLabel('companies',resource.company,en)}<ExternalLink size={12}/></span><b>{en?resource.name_en:resource.name}</b><small className="interview-resource-kind">{resource.kind==='guide'?t('官方指南','Official guide'):t('招聘入口','Careers entry')}</small><p>{en?resource.description_en:resource.description}</p><small>{t('来源核对于','Source checked')} {resource.checked}</small></a>);
 return <section className="interview-resource-panel" aria-labelledby="interview-resources-heading">
  <span className="interview-panel-kicker">{t('从官方资料出发','START WITH OFFICIAL SOURCES')}</span>
  <h2 id="interview-resources-heading" tabIndex={-1}>{company?t(`${interviewLabel('companies',company,false)} 官方资料`,`${interviewLabel('companies',company,true)} resources`):t('官方准备与招聘资料','Official preparation & careers')}</h2>
  <p className="interview-resource-context">{company?t('按当前公司收录，具体面试安排请与招聘团队确认。','Resources for this company; confirm role-specific interview arrangements with the recruiting team.'):t('区分面试指南与招聘入口，从原始来源开始准备。','Start with original sources, with interview guides and careers entries clearly labelled.')}</p>
  <div className="interview-resource-list" id="interview-resource-list">{cards(visible)}</div>
  {!resources.length&&<div className="interview-resource-empty"><p>{t('暂未收录该公司的官方准备资料。可以先按岗位练习，再结合社区面经复盘。','No official preparation resource is listed for this company yet. Start with role-based practice and community experiences.')}</p><button type="button" className="text-button" onClick={()=>focusInterviewSection('interview-preparation-heading')}>{t('查看准备路径','Explore preparation paths')}</button></div>}
  {!company&&resources.length>3&&<button className="interview-resources-toggle" type="button" aria-expanded={expanded} aria-controls={extraId} onClick={()=>setExpanded(value=>!value)}>{expanded?t('收起资料列表','Show fewer resources'):t(`查看全部 ${resources.length} 份资料`,`Explore all ${resources.length} resources`)}<ChevronDown size={14}/></button>}
  {!company&&<div id={extraId} className="interview-resource-list" hidden={!expanded}>{expanded&&cards(resources.slice(3))}</div>}
 </section>;
}
