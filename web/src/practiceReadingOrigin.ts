import {resumePracticeReadingOrigin} from './readingReturn';
import {confirmDiscussionPractice} from './discussionPracticeReturn';

export function resumePracticeOrigin(owner:string,problem:{id:string;number:number},href:string,route:string){
 resumePracticeReadingOrigin(owner,problem,href,route);
 confirmDiscussionPractice(owner,problem,href,route);
}
