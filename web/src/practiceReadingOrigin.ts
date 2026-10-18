import {resumePracticeReadingOrigin,replayPracticeReadingOrigin} from './readingReturn';
import {confirmDiscussionPractice} from './discussionPracticeReturn';

export function resumePracticeOrigin(owner:string,problem:{id:string;number:number}|undefined,href:string,route:string){
 resumePracticeReadingOrigin(owner,problem,href,route);
 confirmDiscussionPractice(owner,problem,href,route);
}

export const replayPracticeOrigin=replayPracticeReadingOrigin;
