import {deferredFeature,FeatureRetry} from './deferredFeature';
import type {LibraryCompanyToolsProps} from './libraryCompanyScope';
/** The filter, status and row actions share one already-loaded feature boundary. */
export const CompanyTools=deferredFeature<LibraryCompanyToolsProps>(()=>import('./LibraryCompanyEvidence').then(module=>({default:module.LibraryCompanyTools})),['',''],{loading:()=>null,failure:(props,retry)=>'problem'in props?null:<FeatureRetry retry={retry}/>});
