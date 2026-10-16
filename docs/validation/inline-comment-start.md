# Start long comment jumps at the beginning

Inline reply bodies can now expand beyond a viewport. Centering an already expanded original reply can place its author, heading and first paragraphs above the visible area. Original-comment, posted-comment and show-more navigation now align the destination row's start, with the same 100px sticky-header clearance used by the discussion reader. Existing focus guards, source-position restoration, draft storage and thread reads are unchanged.

The native React/jsdom card check first fails on the previous centering behavior using a 1400px expanded original and a 768px viewport. It exercises the real reference button, verifies the author/start stays within the simulated viewport below the header, returns to the exact original row position, and preserves editor DOM, selection and both scroll axes without additional requests or storage writes. The fixture loads the actual CSS clearance; short replies and the full existing card-action regressions remain active. This is a deterministic interaction/layout contract check, not browser pixel QA.

Before publication, rerun the full frontend/build gates and the genuine account/session HTTP/SQLite inline-reading flow, including the new native start-alignment assertion.
