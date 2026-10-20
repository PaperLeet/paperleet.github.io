import type {ReactNode} from 'react';
import {AlertCircle} from 'lucide-react';
import './writing-feedback.css';

/** Keep response errors and explicit recovery beside the action that submitted them. */
export default function WritingFeedback({error,errorId,errorClassName='',children}:{error?:string;errorId:string;errorClassName?:string;children?:ReactNode}){
 if(!error&&!children)return null;
 return <div className="writing-feedback">
  {error&&<div id={errorId} className={`social-notice error writing-feedback-error ${errorClassName}`} role="alert" aria-atomic="true"><AlertCircle size={19} aria-hidden="true"/><p>{error}</p></div>}
  {children}
 </div>;
}
