import type {InboxNotificationKind,InboxSnapshot} from './inbox';
export const notificationKinds=['all','reply','like','follow','repost','review'] as const;
export function isNotificationKind(value:unknown):value is InboxNotificationKind{return typeof value==='string'&&notificationKinds.some(kind=>kind===value);}
export function inboxSnapshotPath(kind:InboxNotificationKind,filters:boolean,reviews:boolean,before?:{section:'notifications'|'messages';value:number}){
 const params=new URLSearchParams({limit:'20'});
 if(before)params.set((before.section==='notifications'?'notifications':'conversations')+'_before',String(before.value));
 if(reviews)params.set('reviews','1');
 if(filters)params.set('notification_kind',kind);
 return 'inbox?'+params;
}
export function confirmInboxFilter(snapshot:InboxSnapshot,kind:InboxNotificationKind,filters:boolean,reviews:boolean){
 if(reviews&&snapshot.reviews_included!==true)throw Error('Review notifications could not be confirmed.');
 if(filters&&(snapshot.notification_kind!==kind||kind!=='all'&&snapshot.notifications.some(item=>item.kind!==kind)))throw Error('Notification filter could not be confirmed.');
}
