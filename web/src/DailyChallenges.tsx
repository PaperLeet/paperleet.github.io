import {useEffect,useRef,useState} from 'react';
import {ArrowRight,CalendarDays,Check,ChevronLeft,ChevronRight,Clock3,Flame,RefreshCw,TrendingUp} from 'lucide-react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import {useAccount,socialApi} from './social';
import {challengeMonth,dailyRoute,dailySchedule,dailySelection,readDailySnapshot,utcDay,type DailySnapshot} from './dailyChallengeModel';
import {hostedEdition,hostedAppAvailable,hostedAppHref} from './deployment';
import {signInHref} from './accountNavigation';
import {subscribePracticeChanges} from './practiceChanges';
import {appLocation} from './appLocation';
import type {Problem} from './types';
import './daily-challenges.css';

export default function DailyChallenges({problems,route,problemId,compact=false}:{problems:Problem[];route?:string;problemId?:string;compact?:boolean}) {
  const {t,locale}=useI18n(),{user,loading:accountLoading}=useAccount();
  const [clock,setClock]=useState(Date.now()),[retry,setRetry]=useState(0),[state,setState]=useState<{owner:string;day:string;data?:DailySnapshot;error?:string}>(),[starting,setStarting]=useState(false);
  const [month,setMonth]=useState((route&&dailyRoute(route)?.date||utcDay()).slice(0,7));
  const currentOwner=user?.id||'guest',owner=useRef(currentOwner);owner.current=currentOwner;
  const selection=route?dailyRoute(route):{};
  const localToday=utcDay(new Date(clock)),day=selection?.date||localToday;
  const current=state?.owner===currentOwner&&state.day===day?state:undefined;
  const data=current?.data,today=data?utcDay(new Date(data.server_time)):localToday;
  const selected=data?.challenge||dailySelection(day),problem=problems.find(entry=>entry.id===selected?.problem_id);
  const entry=data?.personal?.entry,personal=data?.personal;
  const viewKey=currentOwner+':'+day,view=useRef(viewKey);view.current=viewKey;
  useEffect(()=>setMonth(day.slice(0,7)),[day]);
  const session=useRef<AbortController|undefined>(undefined);
  useEffect(()=>{const timer=setInterval(()=>setClock(Date.now()),30000);return()=>clearInterval(timer);},[]);
  useEffect(()=>{const refresh=()=>setRetry(value=>value+1);window.addEventListener('focus',refresh);const unsubscribe=subscribePracticeChanges(change=>{if(change.kind==='submission')refresh();});return()=>{window.removeEventListener('focus',refresh);unsubscribe();};},[]);
  useEffect(()=>{
    if(!hostedEdition||accountLoading||problemId&&problemId!==dailySelection(day)?.problem_id)return;
    const controller=new AbortController();session.current=controller;
    setState({owner:currentOwner,day});
    void socialApi<unknown>('challenges/daily?date='+encodeURIComponent(day),undefined,controller.signal).then(raw=>{
      if(controller.signal.aborted)return;
      const snapshot=readDailySnapshot(raw,user?.id);
      if(!snapshot||snapshot.challenge.date!==day)throw Error('Invalid challenge response');
      setState({owner:currentOwner,day,data:snapshot});
    }).catch(()=>{if(!controller.signal.aborted)setState({owner:currentOwner,day,error:t('暂时无法读取挑战和 Rating，请重试。','Challenges and Rating are unavailable. Please retry.')});});
    return()=>controller.abort();
  },[currentOwner,day,retry,accountLoading,locale,problemId]);
  useEffect(()=>{owner.current=currentOwner;view.current=viewKey;setStarting(false);return()=>{session.current?.abort();view.current='';};},[viewKey,currentOwner]);
  async function start(){
    if(starting||!user||!selected||!problem||day!==today)return;
    const expectedOwner=user.id,expectedView=viewKey;setStarting(true);
    try {
      const raw=await socialApi<unknown>('challenges/daily',{expected_user_id:expectedOwner,date:day});
      if(owner.current!==expectedOwner||view.current!==expectedView)return;
      const snapshot=readDailySnapshot(raw,expectedOwner);
      if(!snapshot||snapshot.challenge.date!==day)throw Error('Invalid challenge response');
      setState({owner:expectedOwner,day,data:snapshot});appLocation.hash='#/'+problem.number;
    }catch{if(owner.current===expectedOwner&&view.current===expectedView)setState(previous=>({owner:expectedOwner,day,data:previous?.data,error:t('未能开始计分，请重试。','Could not start the rated challenge. Please retry.')}));}
    finally{if(owner.current===expectedOwner&&view.current===expectedView)setStarting(false);}
  }
  if(!selection)return <section className="daily-challenge"><h1>{t('挑战链接无效','Invalid challenge link')}</h1><AppLink href="#/daily-challenge">{t('返回今日挑战','Open today’s challenge')}</AppLink></section>;
  if(problemId&&problemId!==selected?.problem_id)return null;
  const completed=!!entry?.completed_at,isToday=day===today;
  const seconds=selected?Math.max(0,Math.floor((Date.parse(dailySelection(today)?.ends_at||selected.ends_at)-clock)/1000)):0;
  const remaining=`${String(Math.floor(seconds/3600)).padStart(2,'0')}:${String(Math.floor(seconds/60)%60).padStart(2,'0')}`;
  const practiceLink=problem?'#/'+problem.number:undefined;
  const title=problem?(locale==='en'?problem.translations?.en?.title||problem.title:problem.title):t('正在加载题目…','Loading the problem…');
  const history=personal?.history||[],monthDays=challengeMonth(month);
  const available=monthDays.filter(value=>value>=dailySchedule.starts_on&&value<=today);
  const solved=history.filter(row=>row.day.startsWith(month)&&row.completed_at).length;
  const monthLabel=new Intl.DateTimeFormat(locale==='zh'?'zh-CN':'en-US',{month:'long',year:'numeric',timeZone:'UTC'}).format(new Date(month+'-01T00:00:00Z'));
  const changeMonth=(delta:number)=>{const value=new Date(month+'-01T00:00:00Z');value.setUTCMonth(value.getUTCMonth()+delta);const next=utcDay(value).slice(0,7),date=challengeMonth(next).filter(row=>row>=dailySchedule.starts_on&&row<=today).at(-1);if(date)appLocation.hash='#/daily-challenge?date='+date;};
  const published=available.map(date=>({date,problem:problems.find(row=>row.id===dailySelection(date)?.problem_id),entry:history.find(row=>row.day===date)}));
  const difficulty=(value:string)=>({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[value]||value;
  const ratingRules=<details className="daily-rules"><summary>{t('Rating 计分规则','Rating rules')}</summary><p>{t('初始 1000 分。先开始计分，再于当天 UTC 截止前通过服务端完整判题，每天只加分一次。简单／中等／困难题基准为 900／1200／1500，K=32，至少加 1 分。失败不扣分，重复通过不重复计分。这是个人练习 Rating。','Start at 1000. Begin the rated attempt, then pass the complete server suite before midnight UTC. One award per day; fixed difficulty benchmarks are 900/1200/1500, K=32, minimum gain 1. Failures do not subtract points. Repeated passes add no points. This is a personal practice Rating.')}</p></details>;
  const calendar=<div className="daily-calendar-card">
    <div className="daily-calendar-top"><div><span className="daily-calendar-day">{t('今日','Day')} <b>{Number(today.slice(-2))}</b></span><span className="daily-calendar-timer"><Clock3 size={12}/>{remaining} {t('后更新','left')}</span></div><span className="daily-month-mark" aria-hidden="true"><CalendarDays size={21}/></span></div>
    <div className="daily-month-controls"><button type="button" aria-label={t('上个月','Previous month')} disabled={month<=dailySchedule.starts_on.slice(0,7)} onClick={()=>changeMonth(-1)}><ChevronLeft size={16}/></button><h2>{monthLabel}</h2><button type="button" aria-label={t('下个月','Next month')} disabled={month>=today.slice(0,7)} onClick={()=>changeMonth(1)}><ChevronRight size={16}/></button></div>
    <div className="daily-weekdays" aria-hidden="true">{(locale==='zh'?['日','一','二','三','四','五','六']:['S','M','T','W','T','F','S']).map((label,index)=><span key={index}>{label}</span>)}</div>
    <div className="daily-calendar-grid" role="group" aria-label={t(`${monthLabel} 挑战日历`,`${monthLabel} challenge calendar`)}>
      {Array.from({length:new Date(month+'-01T00:00:00Z').getUTCDay()},(_,index)=><span className="daily-calendar-empty" key={'blank'+index}/>)}
      {monthDays.map(date=>{const record=history.find(row=>row.day===date),done=!!record?.completed_at,enabled=date>=dailySchedule.starts_on&&date<=today;return enabled?<AppLink key={date} href={'#/daily-challenge?date='+date} className="daily-calendar-cell" data-today={date===today||undefined} data-selected={date===day||undefined} data-complete={done||undefined} data-started={!!record&&!done||undefined} aria-current={date===day?'date':undefined} aria-label={`${date} · ${done?t('已完成','Completed'):date===today?t('今日挑战','Today’s challenge'):t('练习','Practice')}`} title={date}><span>{done?<Check size={15}/>:Number(date.slice(-2))}</span></AppLink>:<span key={date} className="daily-calendar-cell unavailable" aria-disabled="true" aria-label={date+' · '+t('尚未开放','Unavailable')}>{Number(date.slice(-2))}</span>;})}
    </div>
    <div className="daily-calendar-legend"><span><i className="complete"/>{t('已完成','Completed')}</span><span><i className="today"/>{t('今天','Today')}</span><span className="daily-calendar-zone">UTC</span></div>
    <div className="daily-month-progress"><span>{t('本月完成','This month')} <b>{personal?solved:'—'}</b><small> / {available.length}</small></span><span>{personal?t(`${personal.completed_days} 天累计`,`${personal.completed_days} days total`):t('登录同步记录','Sign in to sync')}</span></div>
    <div className="daily-month-progress-track" aria-hidden="true"><span style={{width:`${personal&&available.length?solved/available.length*100:0}%`}}/></div>
    {compact&&<AppLink className="daily-calendar-problem" href="#/daily-challenge"><CalendarDays size={15}/><span>{title}</span><ArrowRight size={14}/></AppLink>}
    {compact&&<div className="daily-calendar-account">{personal?<><TrendingUp size={14}/><span>Rating <b>{personal.rating}</b></span></>:hostedAppAvailable?<AppLink href={hostedAppHref(signInHref('#/daily-challenge'))}>{t('登录后参加 Rating','Sign in for Rating')}<ArrowRight size={13}/></AppLink>:<AppLink href="#/daily-challenge">{t('查看每日挑战','View daily challenge')}<ArrowRight size={13}/></AppLink>}</div>}
  </div>;
  if(problemId)return <div className="daily-workspace-note"><CalendarDays size={15}/><span>{t('今日挑战','Today’s challenge')}</span>{completed&&<span className="daily-workspace-complete"><Check size={13}/>+{entry?.delta} Rating</span>}<AppLink href="#/daily-challenge">{t('查看月历','Calendar')}<ArrowRight size={13}/></AppLink></div>;
  if(compact)return <section className="daily-challenge-widget" aria-label={t('每日挑战','Daily challenge')}>{calendar}{current?.error&&<div className="daily-inline-error" role="status"><span>{t('记录暂不可用','Records unavailable')}</span><button type="button" onClick={()=>setRetry(value=>value+1)}>{t('重试','Retry')}</button></div>}</section>;
  return <section className="daily-challenge-page" aria-label={t('每日挑战','Daily challenge')}>
    <header className="daily-page-heading"><div><h1>{t('每日挑战','Daily challenge')}</h1><p>{t('每天一道，把论文里的想法写成代码。','One paper idea. One implementation. Every day.')}</p></div><AppLink href="#">{t('返回题库','Back to problems')}<ArrowRight size={14}/></AppLink></header>
    <div className="daily-page-layout"><div className="daily-page-main">
      <article className="daily-problem-card">
        <div className="daily-problem-top"><span className="daily-problem-eyebrow"><CalendarDays size={15}/>{isToday?t('今日一题','TODAY’S CHALLENGE'):day}</span><span className="daily-problem-date">{day}</span></div>
        <h2>{title}</h2>
        <div className="daily-problem-meta"><span className={`daily-level ${problem?.difficulty||'E'}`}>{difficulty(problem?.difficulty||'E')}</span><span><Clock3 size={13}/>{problem?.estimated_minutes||25} min</span><span>PyTorch</span></div>
        <p className="daily-problem-goal">{problem?t(problem.module.goal,problem.module.goal_en):t('题库加载后可继续练习。','You can practice when the catalog is ready.')}</p>
        {problem?.paper?.title&&<div className="daily-paper-reference"><span>{t('论文 / 来源','PAPER / SOURCE')}</span><p>{problem.paper.title}</p></div>}
        {completed?<div className="daily-complete" role="status"><Check size={17}/><span>{t('当日挑战已完成','Daily challenge completed')}</span><b>+{entry?.delta} Rating</b></div>:null}
        <div className="daily-problem-bottom"><div className="daily-actions">
          {hostedEdition&&user&&isToday&&!completed?<button type="button" className="button" disabled={accountLoading||starting||!problem||!data} onClick={()=>void start()}>{starting?t('正在开始…','Starting…'):entry?t('继续今日挑战','Continue daily challenge'):t('开始计分挑战','Start rated challenge')}<ArrowRight size={15}/></button>:practiceLink?<AppLink className="button" href={practiceLink}>{completed?t('再次练习','Practice again'):t('开始练习','Practice now')}<ArrowRight size={15}/></AppLink>:null}
          {hostedEdition&&user&&isToday&&!completed&&practiceLink&&<AppLink className="daily-secondary-action" href={practiceLink}>{t('不计分练习','Practice only')}</AppLink>}
          {!user&&hostedAppAvailable&&<AppLink className="daily-secondary-action" href={hostedAppHref(signInHref('#/daily-challenge'))}>{t('登录后参加 Rating','Sign in for Rating')}</AppLink>}
        </div><span className="daily-deadline">{isToday?<><Clock3 size={12}/>{remaining} {t('后更新','left')}</>:t('往期练习','Past challenge')}</span></div>
        {entry&&!completed&&<p className="daily-attempt-note">{t('计分已开始，今日截止前提交完整测试即可。','Rated attempt started. Submit the complete suite before today’s deadline.')}</p>}
        {current?.error&&<div className="daily-inline-error" role="alert"><span>{current.error}</span><button type="button" onClick={()=>setRetry(value=>value+1)}>{t('重试','Retry')}</button></div>}
      </article>
      <section className="daily-month-list"><div className="daily-list-heading"><h2>{t('本月题目','This month’s challenges')}</h2><span>{monthLabel}</span></div><div className="daily-list-columns" aria-hidden="true"><span>{t('日期','Date')}</span><span>{t('题目','Problem')}</span><span>{t('难度','Difficulty')}</span><span>{t('状态','Status')}</span></div>
        {!published.length&&<p className="daily-list-empty">{t('本月尚无已开放的挑战。','No published challenges in this month.')}</p>}
        {published.map(row=><AppLink className="daily-list-row" data-current={row.date===day||undefined} href={'#/daily-challenge?date='+row.date} key={row.date}><time>{Number(row.date.slice(-2))}<small>{row.date===today?t('今天','Today'):''}</small></time><span>{row.problem?(locale==='en'?row.problem.translations?.en?.title||row.problem.title:row.problem.title):row.date}</span><span className={`daily-level ${row.problem?.difficulty||'E'}`}>{difficulty(row.problem?.difficulty||'E')}</span><span className="daily-list-status">{row.entry?.completed_at?<Check size={16}/>:row.entry?<span className="daily-started-dot"/>:<span className="daily-status-dash">—</span>}</span></AppLink>)}
        {month===today.slice(0,7)&&<div className="daily-next-note"><Clock3 size={13}/>{t('下一题将在 UTC 00:00 更新','The next challenge arrives at 00:00 UTC')}</div>}
      </section>
    </div><aside className="daily-page-sidebar">{calendar}<section className="daily-rating-card"><header><TrendingUp size={16}/><h2>{t('挑战 Rating','Challenge Rating')}</h2></header><div className="daily-rating-number">{personal?personal.rating:'—'}<span>{personal?t('个人练习评分','Personal practice rating'):t('登录后开启','Sign in to begin')}</span></div><div className="daily-rating-stats"><span><Flame size={14}/>{t('累计完成','Completed')}</span><b>{personal?personal.completed_days:'—'} <small>{t('天','days')}</small></b></div>{ratingRules}<button className="daily-refresh" type="button" disabled={starting} onClick={()=>setRetry(value=>value+1)}><RefreshCw size={12}/>{t('刷新挑战','Refresh challenge')}</button></section></aside></div>
  </section>;
}

export function DailyProblemRow({problems}:{problems:Problem[]}) {
  const {t,locale}=useI18n(),[day,setDay]=useState(utcDay);
  useEffect(()=>{const timer=setInterval(()=>setDay(utcDay()),30000);return()=>clearInterval(timer);},[]);
  const problem=problems.find(row=>row.id===dailySelection(day)?.problem_id);
  if(!problem)return null;
  return <AppLink className="daily-featured-row" href="#/daily-challenge"><span className="daily-featured-icon"><CalendarDays size={18}/></span><span className="daily-featured-label">{t('每日一题','Daily challenge')}</span><strong>{problem.number}. {locale==='en'?problem.translations?.en?.title||problem.title:problem.title}</strong><span className={`daily-level ${problem.difficulty}`}>{({E:t('简单','Easy'),M:t('中等','Medium'),H:t('困难','Hard')} as Record<string,string>)[problem.difficulty]}</span><ArrowRight size={16}/></AppLink>;
}
