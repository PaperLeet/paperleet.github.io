import {browserReason} from "./browserMessages";
import {useI18n} from "./i18n";

export default function BrowserFailureMessage({reason}:{reason:string}){
 const {locale}=useI18n();
 return <>{browserReason(reason,locale === "en")}</>;
}
