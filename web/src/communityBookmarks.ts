export type BookmarkState = { owner_id: string; post_id: string; bookmarked: boolean };

/** A late/wrong acknowledgement must not change another account or post. */
export function confirmedBookmark(value: unknown, owner: string, post: string, expected?: boolean): BookmarkState | undefined {
  if (!owner || !post || !value || typeof value !== 'object' || Array.isArray(value)) return;
  const row = value as Partial<BookmarkState>;
  if (row.owner_id !== owner || row.post_id !== post || typeof row.bookmarked !== 'boolean'
      || expected !== undefined && row.bookmarked !== expected) return;
  return {owner_id:owner,post_id:post,bookmarked:row.bookmarked};
}
