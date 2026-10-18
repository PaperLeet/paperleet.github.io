import {useI18n} from './i18n';
import {validReviewUserMessage} from './reviewUserMessageModel';
import './review-user-message.css';
export default function ReviewUserMessage({message,label}:{message?:string;label?:string}){
 const {t}=useI18n();
 if(!validReviewUserMessage(message)||!message.trim())return null;
 return <details className="review-user-message"><summary>{label||t('查看处理说明','Read the review explanation')}</summary><p>{message}</p></details>;
}
