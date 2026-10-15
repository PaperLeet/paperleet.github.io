import {Avatar,PersonLink,TestAccountBadge,profileHref,type PublicUser} from './social';
import AffiliationBadge from './AffiliationBadge';
import {useI18n} from './i18n';
import './community-identity.css';

function anonymousUser(user:PublicUser,name:string):PublicUser{
  return {...user,display_name:name,avatar_color:'#6b7280',avatar_url:null};
}
export function CommunityAuthorAvatar({user,size=36}:{user:PublicUser;size?:number}){
  const {t}=useI18n();
  return user.anonymous?<span className="anonymous-avatar"><Avatar user={anonymousUser(user,t('匿名用户','Anonymous'))} size={size}/></span>:<a href={profileHref(user)} aria-label={user.display_name}><Avatar user={user} size={size}/></a>;
}
export default function CommunityAuthor({user,compact=false,avatar=true}:{user:PublicUser;compact?:boolean;avatar?:boolean}){
  const {t}=useI18n(),name=t('匿名用户','Anonymous');
  return <span className="community-author-identity">
    {user.anonymous?<span className="person-link anonymous-author">{avatar&&<Avatar user={anonymousUser(user,name)} size={compact?24:34}/>}<span>{name}</span></span>:avatar?<PersonLink user={user} compact={compact}/>:<><a className="post-author-name" href={profileHref(user)}>{user.display_name}</a><TestAccountBadge isTestUser={user.is_test_user}/></>}
    <AffiliationBadge affiliation={user.affiliation} institution={user.institution}/>
  </span>;
}
