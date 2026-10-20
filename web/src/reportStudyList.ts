import type {ProblemTopic} from './communityTopics';
import type {StudyListModule} from './InterviewBatchSave';
import {communityContinuation} from './communityNavigation';
import {readInterviewRoute,interviewHref} from './interviewNavigation';

/** Capture all published links in order, never a silent partial list or workspace data. */
export function captureReportStudyList(questions:readonly string[],topics:readonly ProblemTopic[]){
 if(!questions.length||questions.length>12||new Set(questions).size!==questions.length||questions.some(id=>typeof id!=='string'||id.length>180||!/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(id)))return;
 const modules:StudyListModule[]=[];
 for(const id of questions){
  const matches=topics.filter(topic=>topic.id===id),topic=matches[0];
  if(matches.length!==1||topic.retired||!Number.isSafeInteger(topic.number)||!topic.number||topic.number<1||topic.number>99999||typeof topic.title!=='string'||!topic.title.trim())return;
  modules.push({id,number:topic.number,title:topic.title,...(typeof topic.title_en==='string'&&topic.title_en?{translations:{en:{title:topic.title_en}}}:{})});
 }
 return {ids:[...questions],modules};
}

/** Reply/return-link changes keep the same post; a different source retires the form. */
export function reportStudyListSurface(current:string,source:string){
 const expected=readInterviewRoute(source),view=readInterviewRoute(current);
 if(expected)return !!view&&interviewHref(view)===interviewHref(expected);
 const post=(value:string)=>communityContinuation(value)?.match(/^community\/([a-f0-9]{32})(?:\?|$)/)?.[1];
 const id=post(source);return !!id&&post(current)===id;
}
