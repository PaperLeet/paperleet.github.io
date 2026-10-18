import {interviewQuestionText} from './interviewQuestionDraft';
import {clearPublishedDraft,draftKey,hasDraft,readPostDraft,type DraftStorage,type PostDraft} from './communityDrafts';
import {clearInterviewDraft,interviewDraftKey,isInterviewDraftDurable,readInterviewDraft,type InterviewDraft} from './interviewDrafts';
import {heldPublications,publicationKey,publicationSignature,readPublication,type PendingPublication} from './publicationRequests';

export type InventoryStorage=DraftStorage&Pick<Storage,'length'|'key'>;
export type DraftKind='post'|'solution'|'interview';
export type DraftEntry={id:string;owner:string;kind:DraftKind;scope:string;title:string;body:string;raw:string|null;durable:boolean;post?:PostDraft;interview?:InterviewDraft;pending?:PendingPublication};
export type DraftInventory={rows:DraftEntry[];failed:boolean;limited:boolean;unreadable:number};
const scopes=/^(?:general|[a-z0-9_-]+\/[a-z0-9_-]+)(?::solutions)?$/;
const validScope=(scope:string)=>scope.length<=220&&scopes.test(scope);
const postPrefix=(owner:string)=>'paperleet.community.draft.v1:'+encodeURIComponent(owner)+':';
const requestPrefix=(owner:string)=>'paperleet.community.publication.v1:'+encodeURIComponent(owner)+':post:';
const compare=new Intl.Collator('en',{numeric:true,sensitivity:'base'});

/** Reading a draft inventory never writes, publishes or fetches public/private records. */
export function readDraftInventory(storage:InventoryStorage|undefined,owner:string):DraftInventory{
 const result:DraftInventory={rows:[],failed:!storage,limited:false,unreadable:0};if(!owner)return {...result,failed:false};
 const candidates=new Set<string>(),prefix=postPrefix(owner),requests=requestPrefix(owner);
 if(storage)try{
  const length=storage.length;if(!Number.isSafeInteger(length)||length<0)throw Error('Unreadable storage length');result.limited=length>5000;
  for(let index=0;index<Math.min(length,5000);index++){
   const key=storage.key(index);if(!key)continue;
   const start=key.startsWith(prefix)?prefix:key.startsWith(requests)?requests:undefined;if(!start)continue;
   try{
    const scope=decodeURIComponent(key.slice(start.length));
    if(!validScope(scope)||key!==(start===prefix?draftKey(owner,scope):publicationKey(owner,'post',scope))){result.unreadable++;continue;}
    candidates.add(scope);
   }catch{result.unreadable++;}
  }
 }catch{result.failed=true;}
 const held=heldPublications(owner);for(const row of held)if(row.receipt.kind==='post'&&validScope(row.receipt.scope))candidates.add(row.receipt.scope);
 if(candidates.size>500)result.limited=true;
 const raw=(key:string)=>{try{return storage?.getItem(key)??null;}catch{result.failed=true;return null;}};
 const requestStored=(pending:PendingPublication|undefined)=>{
  if(!pending)return true;
  const {receipt}=pending;
  try{const stored=JSON.parse(raw(publicationKey(owner,receipt.kind,receipt.scope))||'null');return stored?.owner===owner&&stored.kind===receipt.kind&&stored.scope===receipt.scope&&stored.requestId===receipt.requestId&&publicationSignature(stored.payload)===publicationSignature(receipt.payload);}catch{return false;}
 };
 for(const scope of [...candidates].sort(compare.compare).slice(0,500)){
  const source=raw(draftKey(owner,scope)),post=readPostDraft(storage,owner,scope),pending=readPublication(storage,owner,'post',scope);
  if(source&&!post)result.unreadable++;
  if(!post&&!pending){if(raw(publicationKey(owner,'post',scope)))result.unreadable++;continue;}
  if((!post||!hasDraft(post))&&!pending)continue;
  result.rows.push({id:'post:'+scope,owner,kind:scope.endsWith(':solutions')?'solution':'post',scope,title:post?.title??String(pending?.receipt.payload.title||''),body:post?.body??String(pending?.receipt.payload.body||''),raw:post?source:null,durable:(!post||!!source)&&requestStored(pending),...(post?{post}:{}),...(pending?{pending}:{})});
 }
 const interviewRaw=raw(interviewDraftKey(owner)),interview=readInterviewDraft(storage,owner),pending=readPublication(storage,owner,'interview','general');
 let interviewSnapshot=interviewRaw;
 if(interviewRaw)try{const envelope=JSON.parse(interviewRaw);if(envelope?.owner!==owner||envelope.version!==1){interviewSnapshot=null;result.unreadable++;}}catch{interviewSnapshot=null;result.unreadable++;}
 if(interviewRaw&&!interview)result.unreadable++;
 if(!pending&&raw(publicationKey(owner,'interview','general')))result.unreadable++;
 if(interview||pending)result.rows.push({id:'interview:general',owner,kind:'interview',scope:'general',title:interview?.title??String(pending?.receipt.payload.title||''),body:interview?.body??String(pending?.receipt.payload.body||''),raw:interview?interviewSnapshot:null,durable:(!interview||isInterviewDraftDurable(storage,owner,interview))&&requestStored(pending),...(interview?{interview}:{}),...(pending?{pending}:{})});
 const order={post:0,solution:1,interview:2};result.rows.sort((a,b)=>order[a.kind]-order[b.kind]||compare.compare(a.title,b.title)||compare.compare(a.scope,b.scope));
 return result;
}

/** Recheck the exact private snapshot and outstanding request before deleting one draft. */
export function discardInventoryDraft(storage:InventoryStorage|undefined,owner:string,row:DraftEntry):'removed'|'changed'|'pending'|'unavailable'{
 if(!storage)return 'unavailable';
 if(!owner||owner!==row.owner||!['post','solution','interview'].includes(row.kind)||row.id!==(row.kind==='interview'?'interview:general':'post:'+row.scope)||(row.kind==='interview'?row.scope!=='general':!validScope(row.scope)||(row.kind==='solution')!==row.scope.endsWith(':solutions')))return 'changed';
 if(readPublication(storage,owner,row.kind==='interview'?'interview':'post',row.scope))return 'pending';
 try{
  const key=row.kind==='interview'?interviewDraftKey(owner):draftKey(owner,row.scope);
  if(storage.getItem(key)!==row.raw)return 'changed';
  if(row.kind==='interview'){
   if(!row.interview||JSON.stringify(readInterviewDraft(storage,owner))!==JSON.stringify(row.interview))return 'changed';
   return clearInterviewDraft(storage,owner,row.interview)?'removed':'unavailable';
  }
  if(!row.post||JSON.stringify(readPostDraft(storage,owner,row.scope))!==JSON.stringify(row.post))return 'changed';
  return clearPublishedDraft(storage,owner,row.scope,row.post)?'removed':'unavailable';
 }catch{return 'unavailable';}
}

export function matchesInventoryDraft(row:DraftEntry,query:string){
 const normalize=(value:string)=>value.normalize('NFKC').toLocaleLowerCase('en');
 const terms=normalize(Array.from(query).slice(0,200).join('')).trim().split(/\s+/u).filter(Boolean);
 const corpus=normalize([row.title,row.body,interviewQuestionText(row.interview?.questionScratch),row.scope,row.interview?.interview.company,row.interview?.interview.role,row.interview?.interview.location].filter(Boolean).join('\n'));
 return terms.every(term=>corpus.includes(term));
}
