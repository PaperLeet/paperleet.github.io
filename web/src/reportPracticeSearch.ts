import type {ProblemTopic} from './communityTopics';

export const reportPracticeQuery=(value:string)=>Array.from(value).slice(0,100).join('').replace(/[\u0000-\u001f\u007f]/g,' ');
const normalize=(value:string)=>value.normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim();

/** Search only the current report's public IDs and loaded topic labels. */
export function reportPracticeMatches(ids:readonly string[],topics:ReadonlyMap<string,ProblemTopic>,query:string){
 const terms=[...new Set(normalize(reportPracticeQuery(query)).split(' ').filter(Boolean))];
 return ids.filter(id=>{
  const topic=topics.get(id);
  const text=normalize([id,topic?.title||'',topic?.title_en||'',topic?.number?`#${topic.number}`:''].join(' '));
  return terms.every(term=>text.includes(term));
 });
}
