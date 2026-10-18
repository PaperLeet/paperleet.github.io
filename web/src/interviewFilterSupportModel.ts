import {emptyInterviewContext,hasInterviewContext} from './interviewContext';
import type {InterviewFilters} from './interviewNavigation';

type UnsupportedFilterPatch=Partial<Pick<InterviewFilters,'outcome'|'location'|'month_from'|'month_to'>>;
export type InterviewFilterSupportState={kind:'readonly'}|{kind:'ready'}|{kind:'pending';outcome:boolean;context:boolean}|{kind:'unavailable';outcome:boolean;context:boolean;clear:UnsupportedFilterPatch};
/** Capability checks never relax an applied report cohort or rewrite its URL. */
export function interviewFilterSupport(hosted:boolean,filters:InterviewFilters,loading:boolean,capabilities?:Record<string,unknown>):InterviewFilterSupportState{
 if(!hosted)return {kind:'readonly'};
 const outcome=!!filters.outcome,context=hasInterviewContext(filters);
 if(loading&&(outcome||context))return {kind:'pending',outcome,context};
 const missingOutcome=outcome&&capabilities?.interview_outcome_filter!==true,missingContext=context&&capabilities?.interview_context_filters!==true;
 if(!missingOutcome&&!missingContext)return {kind:'ready'};
 return {kind:'unavailable',outcome:missingOutcome,context:missingContext,clear:{...(missingOutcome?{outcome:''}:{}),...(missingContext?emptyInterviewContext():{})}};
}
