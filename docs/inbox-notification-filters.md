# Notification types in the inbox

The notification tab offers All, Replies, Likes, Followers and Reposts. Review outcomes appear when the account service advertises review notifications. Type selection applies to the full stored notification history, including older pages; selecting a type never acknowledges a notification.

Unread badges remain global. “Mark all as read” covers all notification types and direct messages through the captured account-wide cutoffs. The notification-only fallback covers every notification type through its event cutoff. A filtered list explains this scope. Individual read actions still target their captured notification group.

Changing a type, or retrying a failed type request, replaces only the notification window. Already loaded conversations, their pagination cursor and their observed message cutoff remain intact. Background polls acknowledge existing rows and update counts without replacing the reading window. The explicit latest-list action retains its intentional full-refresh behavior.

## API compatibility

`account/me` advertises `inbox_notification_filters: true`. When present, the panel sends `notification_kind=all|reply|like|follow|repost|review` on inbox snapshot and pagination requests. The response echoes `notification_kind`; the client rejects unconfirmed or mixed-type responses. The review type requires `reviews=1`.

Without the capability, the panel uses the existing unfiltered URLs. Omitting the query preserves the legacy DTO without extra filter metadata. Summary and read/write contracts are unchanged; summary requests reject additional filters, and writes do not accept notification types. Type parameters are validated and bound in the grouped SQLite query. Counts, event/message cutoffs and conversation pages are independent of type selection.

## Validation

`web/scripts/check-inbox-notification-filters.mjs` exercises actual React components with controlled requests: same-type paging, rapid changes and obsolete responses, server echo validation, error recovery, older conversation retention, immutable read boundaries, capability changes, owner transitions, native select Escape handling and private code/draft/caret/scroll preservation.

`web/scripts/check-inbox-filters-http.mjs` requires an isolated authenticated loopback fixture. It checks actual React inbox reads against HTTP and SQLite over thirty replies, grouped older likes, follows, reposts and two conversations. It permits no UI writes. Authentication session timestamps may advance; notification, message, community and quota business tables must remain unchanged.

Private backend tests cover strict query validation, grouping and paging beyond the initial page, global counts/cutoffs, anonymous projection, active-actor rules, review opt-in, authentication and cross-account isolation. No physical-browser geometry or production behavior is inferred from these checks.
