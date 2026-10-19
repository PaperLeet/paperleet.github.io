import type {InboxCounts,InboxSnapshot} from './inbox';

/** A background snapshot may acknowledge existing rows, but never replaces the reading window. */
export function inboxListUpdate(current:InboxSnapshot,next:InboxSnapshot) {
  if(current.user_id!==next.user_id)throw Error('Inbox owner changed.');
  const notifications=new Map(next.notifications.map(row=>[row.id,row])),conversations=new Map(next.conversations.map(row=>[row.id,row]));
  const knownNotifications=new Map(current.notifications.map(row=>[row.id,row.latest_event_id]));
  const knownConversations=new Map(current.conversations.map(row=>[row.id,row.last_message.id]));
  const changed=next.notifications.some(row=>knownNotifications.get(row.id)!==row.latest_event_id)
    ||next.conversations.some(row=>knownConversations.get(row.id)!==row.last_message.id);
  const counts:InboxCounts={user_id:next.user_id,unread_count:next.unread_count,notification_unread_count:next.notification_unread_count,message_unread_count:next.message_unread_count};
  return {changed,snapshot:{...current,...counts,
    notifications:current.notifications.map(row=>{const fresh=notifications.get(row.id);return fresh?.latest_event_id===row.latest_event_id&&fresh.unread!==row.unread?{...row,unread:fresh.unread}:row;}),
    conversations:current.conversations.map(row=>{const fresh=conversations.get(row.id);return fresh?.last_message.id===row.last_message.id&&fresh.unread_count!==row.unread_count?{...row,unread_count:fresh.unread_count}:row;})}};
}

/** A notification filter replaces only that window; older direct messages stay put. */
export function replaceInboxNotifications(current:InboxSnapshot,next:InboxSnapshot){
 const update=inboxListUpdate(current,{...next,notifications:current.notifications});
 return {changed:update.changed,snapshot:{...update.snapshot,notifications:next.notifications,
  notifications_next_before:next.notifications_next_before,observed_event_id:next.observed_event_id,
  notification_kind:next.notification_kind,reviews_included:next.reviews_included}};
}
