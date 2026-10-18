import {useId} from 'react';
import {useI18n} from './i18n';
import {MAX_REVIEW_USER_MESSAGE,validReviewUserMessage} from './reviewUserMessageModel';
import './review-user-message.css';
export default function ModerationUserMessage({value,onChange,remove,postAvailable,draftOnly=false}:{value:string;onChange:(value:string)=>void;remove:boolean;postAvailable:boolean;draftOnly?:boolean}){
 const {t}=useI18n(),id=useId(),valid=validReviewUserMessage(value),count=Array.from(value).length;
 const recipients=remove&&postAvailable?t('作者与本批待审举报人','Author and pending reporters'):t('本批待审举报人','Pending reporters');
 return <div className="moderation-user-message-field"><label htmlFor={id}>{t('给用户的说明','Message to users')} · {draftOnly?t('仅本页草稿','Unsent draft'):remove?t('必填','Required'):t('选填','Optional')}</label><textarea id={id} rows={3} maxLength={MAX_REVIEW_USER_MESSAGE*2} required={remove&&!draftOnly} value={value} aria-invalid={!valid||undefined} aria-describedby={`${id}-hint`} onChange={event=>onChange(event.target.value)}/>
  <div id={`${id}-hint`} className="moderation-user-message-hint"><span>{t('仅这段内容会随处理结果发送；举报身份、证据与内部备注请勿填入这里。','Only this explanation accompanies the outcome. Keep reporter identities, evidence and internal notes out of this field.')}</span><span aria-label={t('说明字数','Explanation length')}>{count} / {MAX_REVIEW_USER_MESSAGE}</span></div>
  {!valid&&<p role="alert">{t(`请将说明控制在 ${MAX_REVIEW_USER_MESSAGE} 个字符以内，且不含空字符。`,`Use up to ${MAX_REVIEW_USER_MESSAGE} characters without null characters.`)}</p>}
  <section className="moderation-user-message-preview" aria-label={draftOnly?t('未提交草稿预览','Unsent draft preview'):t('发送预览','Delivery preview')}><b>{draftOnly?t('未提交草稿预览','Unsent draft preview'):t('发送预览','Delivery preview')}</b> · {draftOnly?t('后续输入不会再次发送','Later input is not delivered'):recipients}<p>{value.trim()||t('填写简短说明，帮助用户理解这次处理。','Add a brief explanation to help users understand this decision.')}</p></section>
 </div>;
}
