import {ArrowUpRight,FileCheck,Flag,Layers} from 'lucide-react';
import {DateLabel} from './social';
import {useI18n} from './i18n';
import {adminHref} from './adminLinks';
import {ReadStatus,useAdminRead} from './AdminParts';
import {isAdminReviewInbox,type AdminReviewInboxData} from './adminInboxModel';

export default function AdminReviewInbox({owner,epoch}:{owner:string;epoch:number}){
 const {t}=useI18n(),result=useAdminRead<AdminReviewInboxData>('admin/inbox',owner,epoch,isAdminReviewInbox);
 const fresh=!result.loading&&!result.error;
 const groups=[
  {key:'reports',Icon:Flag,title:t('待审举报','Reported posts'),hint:t('优先处理独立举报人数多的帖子','Most pending reporters first'),href:adminHref({section:'reports',page:1})},
  {key:'intake',Icon:FileCheck,title:t('论文初审','Paper intake'),hint:t('先处理较早提交的论文','Oldest submissions first'),href:adminHref({section:'papers',page:1})},
  {key:'candidates',Icon:Layers,title:t('候选版本审核','Candidate review'),hint:t('审核当前版本，批准不等于发布','Review the current version; approval precedes publication'),href:adminHref({section:'papers',status:'all',review:'pending',page:1})},
 ] as const;
 return <section className="admin-review-inbox" aria-labelledby="admin-review-inbox-title" aria-busy={result.loading}>
  <div className="admin-inbox-heading"><div><h2 id="admin-review-inbox-title">{t('优先处理','Review next')}</h2><p>{t('每组预览最多 3 项，点击进入后核对完整内容与证据。','Preview up to three items per queue. Open an item to review its full content and evidence.')}</p></div></div>
  <ReadStatus {...result}/>
  {result.data&&<div className="admin-inbox-grid">{groups.map(({key,Icon,title,hint,href})=>{
   const group=result.data![key];
   return <section className="admin-inbox-group" key={key} aria-label={title}>
    <header><Icon size={19} aria-hidden="true"/><h3>{title}</h3><span className="admin-inbox-total">{group.total}</span></header><p className="admin-inbox-hint">{hint}</p>
    {!group.items.length?<p className="admin-inbox-empty">{t('这一组已处理完毕','Nothing awaiting review')}</p>:<ul>{key==='reports'?result.data!.reports.items.map(row=>{
     const content=<><b>{row.title}</b><span className="admin-inbox-meta"><span>{t(`${row.pending_reports} 人待审 · 累计 ${row.total_reports} 人举报`,`${row.pending_reports} pending · ${row.total_reports} total reporters`)}</span><DateLabel value={row.last_report_at}/></span>{!row.post_available&&<small>{t('原帖已不存在，举报快照仍可审核','Post unavailable; report snapshots remain available')}</small>}</>;
     return <li key={row.post_id}>{fresh?<a href={adminHref({section:'reports',page:1,item:row.post_id})}>{content}<ArrowUpRight size={16} aria-hidden="true"/></a>:<div className="admin-inbox-row">{content}</div>}</li>;
    }):result.data![key].items.map(row=>{
     const content=<><b>arXiv: {row.arxiv_id}</b><span className="admin-inbox-meta"><span>{key==='intake'?t('等待初审','Awaiting intake review'):t(`候选版本 v${row.version}`,`Candidate v${row.version}`)}</span><DateLabel value={row.created_at}/></span></>;
     return <li key={row.id}>{fresh?<a href={adminHref({section:'papers',page:1,status:'all',item:row.id})}>{content}<ArrowUpRight size={16} aria-hidden="true"/></a>:<div className="admin-inbox-row">{content}</div>}</li>;
    })}</ul>}
    {fresh&&<a className="admin-inbox-all" href={href}>{t('查看完整队列','View full queue')}<ArrowUpRight size={15} aria-hidden="true"/></a>}
   </section>;
  })}</div>}
 </section>;
}
