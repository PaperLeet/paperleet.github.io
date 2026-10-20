import {useEffect,useId,useState,type RefObject} from 'react';
import {useAccount,type AccountUser} from './social';
import {useI18n} from './i18n';
import AffiliationBadge from './AffiliationBadge';
import './community-identity.css';

type Props={anonymous:boolean;onChange:(value:boolean)=>void;disabled:boolean;id?:string;containerRef?:RefObject<HTMLDivElement|null>};
export default function PublishingIdentity(props:Props){
  const {user,capabilities}=useAccount();
  return user?<IdentityChoice key={user.id} {...props} user={user} available={capabilities?.community_anonymous_posts===true}/>:null;
}

function IdentityChoice({anonymous,onChange,disabled,id,containerRef,user,available}:Props&{user:AccountUser;available:boolean}){
  const {t}=useI18n(),description=useId();
  // Keep an already offered choice readable and focusable for this account's editor.
  const [choiceShown,setChoiceShown]=useState(available||anonymous);
  useEffect(()=>{if(available||anonymous)setChoiceShown(true);},[available,anonymous]);
  const showChoice=choiceShown||available||anonymous,unavailable=!available&&!anonymous;
  return <div className="publishing-identity" id={id} ref={containerRef}>
    <div className="publishing-identity-preview"><small>{t('发布身份','Posting as')}</small><span className="publishing-identity-name">{anonymous?t('匿名用户','Anonymous'):user.display_name}</span><AffiliationBadge affiliation={user.affiliation} institution={user.institution}/>
      {showChoice&&<label data-unavailable={unavailable||undefined}><input type="checkbox" checked={anonymous} disabled={disabled} aria-disabled={unavailable||undefined} aria-describedby={description} onChange={event=>{if(!disabled&&!unavailable)onChange(event.target.checked);}}/>{t('匿名发帖','Post anonymously')}</label>}
    </div>
    {showChoice&&<p id={description}>{!available?(anonymous?t('匿名发布暂不可用。草稿已保留，请刷新账号后重试，或主动取消匿名。','Anonymous posting is unavailable. Your draft is kept. Refresh your account and retry, or choose named posting.'):t('当前使用昵称发布；匿名选项暂不可用。','Your public name is selected. Anonymous posting is unavailable.')):t('匿名时隐藏昵称、头像和主页链接，所属机构仍公开；该帖下你自己的回复也保持匿名。发布后不能更改身份方式。','Anonymous posts hide your name, avatar and profile link. Affiliation stays public, including on your follow-up replies. Posting identity cannot change after publication.')}</p>}
  </div>;
}
