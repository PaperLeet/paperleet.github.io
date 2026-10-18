import {useMemo,useState} from 'react';
import {ArrowLeft,ArrowUpRight,BookOpen,Search} from 'lucide-react';
import {useI18n} from './i18n';
import type {Paper,Problem} from './types';
import {searchPaperRegistry} from './paperRegistrySearch';
import './paper-registry.css';

export default function PaperRegistry({papers,problems,error,onRetry}:{papers:Paper[];problems:Problem[];error?:string;onRetry:()=>void}){
  const {t,locale}=useI18n(),[query,setQuery]=useState('');
  const entries=useMemo(()=>searchPaperRegistry(papers,problems,query),[papers,problems,query]);
  const total=papers.filter(paper=>paper.kind!=='course').length,byId=new Map(problems.map(problem=>[problem.id,problem]));
  return <main className="social-page paper-registry-page">
    <a className="text-button" href="#"><ArrowLeft size={16}/>{t('返回题库','Back to practice')}</a>
    <header className="paper-registry-heading"><div><h1>{t('论文库','Paper registry')}</h1><p>{t('查看已收录的论文及对应题目，投稿前先查一下。','Browse published papers and their exercises before submitting a paper.')}</p></div><a className="button subtle" href="#/submit">{t('提交论文','Submit paper')}<ArrowUpRight size={16}/></a></header>
    <form role="search" className="paper-registry-search" onSubmit={event=>event.preventDefault()}><Search size={18}/><input type="search" value={query} maxLength={300} onChange={event=>setQuery(event.target.value)} aria-label={t('搜索已收录论文','Search published papers')} placeholder={t('论文名、算法、作者或 arXiv 链接…','Paper, algorithm, author or arXiv link…')}/></form>
    {error?<p role="alert">{t('论文库暂时不可用。','The paper registry is unavailable.')} <button className="text-button" onClick={onRetry}>{t('重试','Retry')}</button></p>:!papers.length?<p role="status">{t('正在加载论文库…','Loading papers…')}</p>:<>
      <p className="paper-registry-count" role="status">{query?t(`${entries.length} / ${total} 篇论文`,`${entries.length} of ${total} papers`):t(`${total} 篇已收录论文`,`${total} published papers`)}</p>
      {!entries.length?<div className="paper-registry-empty"><BookOpen size={24}/><p>{t('没有找到匹配的已收录论文。提交时还会检查候选题中的重复投稿。','No published paper matches. Submission also checks existing proposals.')}</p><a href="#/submit">{t('去提交论文','Submit a paper')}</a></div>:<ol className="paper-registry-list">{entries.map(paper=><li key={paper.id}>
        <div className="paper-registry-meta"><span>#{paper.number} · {paper.short_title}</span><span>{[paper.year,paper.venue].filter(Boolean).join(' · ')}</span></div>
        <h2><a href={`#/papers/${paper.id}`}>{paper.title}</a></h2><p className="paper-registry-authors">{Array.isArray(paper.authors)?paper.authors.join(', '):paper.authors}</p>
        <div className="paper-registry-links"><a href={paper.url} target="_blank" rel="noopener noreferrer">{paper.arxiv?`arXiv:${paper.arxiv}`:t('原文','Source')}<ArrowUpRight size={13}/></a>{paper.module_ids.flatMap(id=>{const p=byId.get(id);return p?[<a key={id} href={`#/${p.number}`}>#{p.number} {locale==='en'?p.translations?.en?.title||p.title:p.title}</a>]:[];})}</div>
      </li>)}</ol>}
    </>}
  </main>;
}
