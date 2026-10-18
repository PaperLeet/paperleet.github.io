import {useId} from 'react';
import {ArrowRight,MessageCircle} from 'lucide-react';
import AppLink from './AppLink';
import {useAccount} from './social';
import {useI18n} from './i18n';
import {signInHref} from './accountNavigation';
import './community-join.css';

export default function CommunityJoinPrompt({returnTo='community'}:{returnTo?:string}){
 const {capabilities,loading}=useAccount(),{t}=useI18n(),id=useId();
 const testAuth=!loading&&capabilities?.auth_mode==='test';
 const message=loading?t('正在确认账号服务，你也可以先前往登录。','Checking account services. You can still open sign-in.')
  :testAuth?t('使用管理员提供的内测账号加入讨论。','Join the discussion with your administrator-issued test account.')
  :capabilities?.community_open_publishing===true?t('发帖和回复无需学校／企业认证，机构归属认证为选填。','School/company verification is optional for posting and replying.')
  :t('登录后返回当前页面，继续浏览或参与讨论。','Return here after signing in to keep browsing or join the conversation.');
 return <section className="join-prompt" aria-labelledby={id}>
  <span className="community-join-icon" aria-hidden="true"><MessageCircle size={22}/></span>
  <div className="community-join-content">
   <div className="community-join-copy"><strong id={id}>{testAuth?t('加入内测讨论','Join the beta discussion'):t('登录，继续探索与交流','Sign in to keep exploring')}</strong><p role={loading?'status':undefined}>{message}</p></div>
   <AppLink className="button" href={signInHref(returnTo)}>{testAuth?t('内测账号登录','Test account sign-in'):t('登录 / 注册','Sign in / Join')}<ArrowRight size={15} aria-hidden="true"/></AppLink>
  </div>
 </section>;
}
