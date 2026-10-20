import {ArrowUpRight} from 'lucide-react';
import {useI18n} from './i18n';
import './edit-review-action.css';

export default function EditReviewAction({hintId,controls,disabled,onReview}:{hintId:string;controls:string;disabled:boolean;onReview:(opener:HTMLButtonElement)=>void}){
 const {t}=useI18n();
 return <div className="edit-review-action"><p id={hintId}>{t('草稿基于较早版本，保存前请先核对最新内容。','Your draft uses an earlier version. Review the latest content before saving.')}</p><button type="button" disabled={disabled} aria-controls={controls} onClick={event=>onReview(event.currentTarget)}>{t('查看最新版本','Review latest version')}<ArrowUpRight size={16} aria-hidden="true"/></button></div>;
}
