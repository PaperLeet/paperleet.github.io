import {createContext,useContext,useMemo,useState,type ReactNode} from 'react';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import type {ProblemTopic} from './communityTopics';
import {practiceMentionIndex,transformPracticeMentions,type MentionIndex} from './practiceMentionMatcher';
import {withLibraryContext} from './libraryNavigation';
import {useI18n} from './i18n';
import './practice-mentions.css';

const Context=createContext<{index?:MentionIndex;returnTo:string}|undefined>(undefined);
export const usePracticeMentions=()=>useContext(Context);
const parser=unified().use(remarkParse).use(remarkGfm).use(remarkMath);

export default function PracticeMentions({topics=[],papers=[],body,returnTo,children}:{topics?:ProblemTopic[];papers?:import('./types').Paper[];body:string;returnTo:string;children:ReactNode}){
 const {t,locale}=useI18n(),[expanded,setExpanded]=useState(false);
 const index=useMemo(()=>{
   const byModule=new Map(papers.filter(paper=>paper.kind!=='course').flatMap(paper=>paper.module_ids.map(id=>[id,paper] as const)));
   return practiceMentionIndex(topics.map(topic=>({...topic,paper:byModule.get(topic.id)||topic.paper,paper_short_title:byModule.get(topic.id)?.short_title})));
 },[topics,papers]);
 const matches=useMemo(()=>{
   const found=new Map<string,ProblemTopic>();
   transformPracticeMentions(parser.parse(body),index,topic=>withLibraryContext(topic.id,returnTo),topic=>found.set(topic.id,topic));
   return [...found.values()];
 },[body,index,returnTo]);
 const context=useMemo(()=>({index,returnTo}),[index,returnTo]);
 return <Context.Provider value={context}>{children}{matches.length>0&&<div className="practice-mentions" aria-label={t('文中识别到的题目','Exercises mentioned in this article')}>
   <span>{t('文中题目','Mentioned exercises')}</span>{(expanded?matches:matches.slice(0,5)).map(topic=><a key={topic.id} href={withLibraryContext(topic.id,returnTo)} data-reading-link="practice" data-reading-practice={topic.id}>{topic.number?`#${topic.number} `:''}{locale==='en'?topic.title_en||topic.title:topic.title}</a>)}
   {matches.length>5&&<button type="button" className="text-button" aria-expanded={expanded} onClick={()=>setExpanded(value=>!value)}>{expanded?t('收起','Show less'):t(`展开其余 ${matches.length-5} 道`,`Show ${matches.length-5} more`)}</button>}
 </div>}</Context.Provider>;
}
