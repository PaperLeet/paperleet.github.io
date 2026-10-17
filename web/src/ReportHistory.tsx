import {useEffect,useRef,useState} from 'react';
import {ArrowLeft,ArrowUpRight,CheckCircle2,Clock3,Flag,RefreshCw} from 'lucide-react';
import {DateLabel,errorMessage,socialApi,useAccount} from './social';
import {useI18n} from './i18n';
import {signInHref} from './accountNavigation';
import {communityPostHref} from './communityNavigation';
import {readReportHistory,reportHistoryHref,MAX_REPORT_HISTORY_PAGE,REPORT_HISTORY_PAGE_SIZE,type ReportHistoryState} from './reportHistoryNavigation';
import {reportReasons,type ReportHistory as History} from './communityReports';
import {useReadingPosition} from './useReadingPosition';
import ReadingReturnNotice from './ReadingReturnNotice';
import './report-history.css';

export default function ReportHistory({route}:{route:string}){
 const {user,loading,capabilities}=useAccount(),{t}=useI18n(),filters=readReportHistory(route);
 if(!filters)return <main className="report-history"><h1>{t('举报记录地址无效','Invalid report-history address')}</h1><a href={reportHistoryHref()}>{t('打开我的举报','Open my reports')}</a></main>;
 const href=reportHistoryHref(filters.state,filters.page);
 if(loading)return <main className="report-history"><p role="status">{t('正在确认账号…','Checking account…')}</p></main>;
 if(!user)return <main className="report-history"><h1>{t('我的举报','My reports')}</h1><p>{t('登录后查看自己提交的举报与处理状态。','Sign in to see your reports and their review status.')}</p><a className="button" href={signInHref(href)}>{t('登录查看举报记录','Sign in to view reports')}</a></main>;
 if(user.status&&user.status!=='active'||capabilities?.community_report_history!==true)return <main className="report-history"><h1>{t('我的举报','My reports')}</h1><p>{t('此账号或服务暂不支持查看举报记录。','Report history is unavailable for this account or service.')}</p></main>;
 return <HistoryView key={user.id+':'+href} owner={user.id} state={filters.state} page={filters.page} href={href}/>;
}
function parseHistory(value:History,owner:string,state:ReportHistoryState,offset:number){
 const counts=value?.summary,validDate=(v:unknown)=>typeof v==='string'&&Number.isFinite(Date.parse(v));
 if(!value||value.owner_id!==owner||value.state!==state||value.offset!==offset||value.limit!==REPORT_HISTORY_PAGE_SIZE||!counts||Object.values(counts).some(n=>!Number.isSafeInteger(n)||n<0)||counts.all!==counts.pending+counts.reviewed||value.total!==counts[state]||!Array.isArray(value.items)||value.items.length>REPORT_HISTORY_PAGE_SIZE||value.items.length!==Math.min(REPORT_HISTORY_PAGE_SIZE,Math.max(0,value.total-offset))||new Set(value.items.map(row=>row?.id)).size!==value.items.length||value.items.some(row=>!row||!/^[a-f0-9]{32}$/.test(row.id)||!/^[a-f0-9]{32}$/.test(row.post_id)||typeof row.title!=='string'||!row.title||row.title.length>180||typeof row.details!=='string'||row.details.length>1000||!reportReasons.some(([reason])=>reason===row.reason)||!validDate(row.created_at)||!['pending','reviewed'].includes(row.status)||state!=='all'&&row.status!==state||typeof row.post_available!=='boolean'||row.reviewed_at!==null&&!validDate(row.reviewed_at)||row.status==='pending'&&row.reviewed_at!==null))throw Error('Your report history could not be confirmed. Retry loading it.');
 return value;
}
function HistoryView({owner,state,page,href}:{owner:string;state:ReportHistoryState;page:number;href:string}){
 const {t}=useI18n(),[data,setData]=useState<History>(),[loading,setLoading]=useState(false),[error,setError]=useState(''),mounted=useRef(true),sequence=useRef(0),heading=useRef<HTMLHeadingElement>(null);
 const inflight=useRef<{version:number;controller:AbortController;promise:Promise<void>}>(undefined),offset=(page-1)*REPORT_HISTORY_PAGE_SIZE;
 const reading=useReadingPosition('account:'+owner,href,!!data&&!error);
 function load(){
  if(inflight.current)return inflight.current.promise;
  const version=++sequence.current,controller=new AbortController();setLoading(true);setError('');
  const timer=setTimeout(()=>controller.abort(),20_000);
  const promise=Promise.resolve().then(async()=>{try{
   const response=await socialApi<History>(`account/community-reports?state=${state}&limit=${REPORT_HISTORY_PAGE_SIZE}&offset=${offset}`,undefined,controller.signal),value=parseHistory(response,owner,state,offset);
   if(mounted.current&&version===sequence.current)setData(value);
  }catch(e){if(mounted.current&&version===sequence.current)setError(controller.signal.aborted?t('读取举报记录超时，当前内容已保留。','Loading report history timed out. Current content is kept.'):errorMessage(e));}
  finally{clearTimeout(timer);if(inflight.current?.version===version)inflight.current=undefined;if(mounted.current&&version===sequence.current)setLoading(false);}});
  inflight.current={version,controller,promise};return promise;
 }
 useEffect(()=>{
  mounted.current=true;void load();
  const refresh=()=>{if(document.visibilityState==='visible')void load();};window.addEventListener('focus',refresh);document.addEventListener('visibilitychange',refresh);
  return()=>{mounted.current=false;sequence.current++;inflight.current?.controller.abort();inflight.current=undefined;window.removeEventListener('focus',refresh);document.removeEventListener('visibilitychange',refresh);};
 },[]);
 const labels={all:t('全部','All'),pending:t('待审核','Pending'),reviewed:t('已处理','Reviewed')};
 return <main ref={reading.container} className="report-history"><header className="report-history-heading"><div><span><Flag size={16} aria-hidden="true"/>{t('仅自己可见','Only visible to you')}</span><h1 ref={heading} tabIndex={-1}>{t('我的举报','My reports')}</h1><p>{t('查看自己提交的举报和处理状态。其他人的新举报不会改变你这次举报的历史状态。','Track your reports and their review status. New reports from others do not change the status of your earlier report.')}</p></div><a href="#/community"><ArrowLeft size={15}/>{t('返回社区','Back to community')}</a></header>
  <div className="report-history-toolbar"><nav aria-label={t('举报状态','Report status')}>{(['all','pending','reviewed'] as const).map(filter=><a key={filter} href={reportHistoryHref(filter)} aria-current={state===filter?'page':undefined}>{labels[filter]}{data&&<span>{data.summary[filter]}</span>}</a>)}</nav><button type="button" className="button subtle" disabled={loading} onClick={()=>void load()}><RefreshCw size={15}/>{loading?t('正在刷新…','Refreshing…'):t('刷新记录','Refresh reports')}</button></div>
  <ReadingReturnNotice notice={reading.notice} onRestore={reading.restore}/>
  {loading&&<p className="report-history-load" role="status">{t('正在读取最新记录…','Loading the latest reports…')}</p>}
  {error&&<div className="report-history-error" role="status"><p>{t('举报记录暂时无法读取，已保留本页内容。','Report history could not load. This page’s content is kept.')} {error}</p><button type="button" className="text-button" disabled={loading} onClick={()=>{heading.current?.focus({preventScroll:true});void load();}}>{t('重试加载记录','Retry report history')}</button></div>}
  <section data-reading-list="" tabIndex={-1} aria-label={t('我的举报记录','My report history')} className="report-history-list">
   {data&&!data.items.length&&<div className="report-history-empty"><Flag size={28} aria-hidden="true"/><h2>{data.total?t('这一页暂无记录','No reports on this page'):state==='all'?t('还没有提交过举报','No reports submitted yet'):state==='pending'?t('没有待审核的举报','No pending reports'):t('还没有已处理的举报','No reviewed reports yet')}</h2><p>{t('举报状态可能已更新。可切换筛选或回到第一页查看。','Review status may have changed. Choose another filter or return to the first page.')}</p>{page>1?<a href={reportHistoryHref(state)}>{t('回到第一页','Back to the first page')}</a>:<a href="#/community">{t('浏览社区','Browse the community')}</a>}</div>}
   {data?.items.map(row=>{const reason=reportReasons.find(([value])=>value===row.reason)!;return <article key={row.id} data-reading-post={row.post_id} className="report-history-card"><div className="report-history-card-heading"><span className={'report-history-status '+row.status}>{row.status==='reviewed'?<CheckCircle2 size={15}/>:<Clock3 size={15}/>} {labels[row.status]}</span><span><DateLabel value={row.created_at}/></span></div><h2>{row.post_available?<a data-reading-link="title" href={communityPostHref(row.post_id,href)}>{row.title}<ArrowUpRight size={15} aria-hidden="true"/></a>:row.title}</h2><p className="report-history-reason">{t('举报原因','Reason')} · {t(reason[1],reason[2])}</p>{row.details&&<details className="report-history-details"><summary>{t('查看我的补充说明','View my details')}</summary><p>{row.details}</p></details>}<footer>{row.status==='reviewed'?<span>{t('已完成人工审核','Human review completed')}{row.reviewed_at&&<> · <DateLabel value={row.reviewed_at}/></>}</span>:<span>{t('已提交，等待人工审核','Submitted and awaiting human review')}</span>}{!row.post_available&&<span>{t('原帖已不可用，举报记录仍保留','Original post unavailable; your report record is retained')}</span>}</footer></article>;})}
  </section>
  {data&&<nav className="report-history-pagination" aria-label={t('举报记录翻页','Report history pages')}>{page>1?<a className="button subtle" href={reportHistoryHref(state,page-1)}>{t('上一页','Previous')}</a>:<span/>}<span>{t(`第 ${page} 页，共 ${data.total} 条`,`Page ${page} · ${data.total} reports`)}</span>{offset+REPORT_HISTORY_PAGE_SIZE<data.total&&page<MAX_REPORT_HISTORY_PAGE?<a className="button subtle" href={reportHistoryHref(state,page+1)}>{t('下一页','Next')}</a>:<span/>}</nav>}
 </main>;
}
