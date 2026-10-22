import AppLink from './AppLink';
import {useI18n} from './i18n';

type Props={conflict:boolean;local:boolean;saving:boolean;health:boolean;signedIn:boolean;testAuth:boolean;signIn:string;onResolve:(keep:boolean)=>void;onRefresh:()=>void};

/** Result-panel notices for a draft conflict or unavailable server judging; they load with practice, not the library shell. */
export default function WorkspaceAccessNotice({conflict,local,saving,health,signedIn,testAuth,signIn,onResolve,onRefresh}:Props) {
  const {t}=useI18n();
  return <div className="practice-notice" role="status">
    {conflict ? (
      <>
        <span>
          {t(
            local ? "另一标签页更新了这道题的草稿。请选择保留当前代码或使用另一版本；也可先下载当前代码。" : "其他设备更新了这道题的草稿。选择保留哪个版本；当前代码尚未覆盖服务器。",
            local ? "Another tab updated this draft. Keep your current code or use the other version; you can download your current code first." : "Another device updated this draft. Choose which version to keep; your current code has not overwritten the server.",
          )}
        </span>
        <button disabled={saving} onClick={() => onResolve(false)}>
          {local ? t('使用另一版本','Use the other version') : t("使用服务器版本", "Use server draft")}
        </button>
        <button disabled={saving} onClick={() => onResolve(true)}>
          {t("保留当前版本", "Keep this draft")}
        </button>
      </>
    ) : !health ? (
      <>
        <span>{t("无法确认服务配置，工作区尚未加载。", "Service configuration is unavailable. Your workspace has not been loaded.")}</span>
        <button onClick={onRefresh}>{t("重试", "Retry")}</button>
      </>
    ) : !signedIn ? (
      <>
        <span>
          {testAuth
            ? t("使用管理员提供的内测账号和访问码登录后，即可运行测试。", "Sign in with your administrator-issued test account and access code to run tests.")
            : t("验证学校或公司邮箱并登录后，即可使用在线判题。", "Verify your school or company email and sign in to use online judging.")}
        </span>
        <AppLink href={signIn}>{t("登录以运行", "Sign in to run")}</AppLink>
      </>
    ) : (
      <>
        <span>{t("判题沙箱尚未就绪，请稍后重试。草稿仍会同步。", "The judge sandbox is not ready. Please try again shortly; your drafts will still sync.")}</span>
        <button onClick={onRefresh}>{t("刷新状态", "Refresh status")}</button>
      </>
    )}
  </div>;
}
