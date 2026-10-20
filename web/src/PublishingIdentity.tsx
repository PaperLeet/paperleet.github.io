import {useId,type RefObject} from 'react';
import {useAccount} from './social';
import {useI18n} from './i18n';
import AffiliationBadge from './AffiliationBadge';
import './community-identity.css';

export default function PublishingIdentity({anonymous,onChange,disabled,id,containerRef}:{anonymous:boolean;onChange:(value:boolean)=>void;disabled:boolean;id?:string;containerRef?:RefObject<HTMLDivElement|null>}){
  const {t}=useI18n(),{user,capabilities}=useAccount(),description=useId();
  if(!user)return null;
  const available=capabilities?.community_anonymous_posts===true;
  return <div className="publishing-identity" id={id} ref={containerRef}>
    <div className="publishing-identity-preview"><small>{t('发布身份','Posting as')}</small><span className="publishing-identity-name">{anonymous?t('匿名用户','Anonymous'):user.display_name}</span><AffiliationBadge affiliation={user.affiliation} institution={user.institution}/>
      {(available||anonymous)&&<label><input type="checkbox" checked={anonymous} disabled={disabled||!available&&!anonymous} aria-describedby={description} onChange={event=>onChange(event.target.checked)}/>{t('匿名发帖','Post anonymously')}</label>}
    </div>
    {(available||anonymous)&&<p id={description}>{anonymous&&!available?t('匿名发布暂不可用。草稿已保留，请刷新账号后重试，或主动取消匿名。','Anonymous posting is unavailable. Your draft is kept. Refresh your account and retry, or choose named posting.') :t('匿名时隐藏昵称、头像和主页链接，所属机构仍公开；该帖下你自己的回复也保持匿名。发布后不能更改身份方式。','Anonymous posts hide your name, avatar and profile link. Affiliation stays public, including on your follow-up replies. Posting identity cannot change after publication.')}</p>}
  </div>;
}
