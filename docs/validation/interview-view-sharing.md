# Copy an applied interview view

The interview hub now offers **Copy filtered link / 复制筛选链接** beside the
existing study and publishing actions. It copies the applied route: company,
role, stage, period, search, sorting, page, concept, curriculum, saved scope and
the active question-view learning state or practice queue. Text typed into the
search field becomes part of that route only after the existing search action.

The URL uses the current HTTP(S) origin and deployment path plus the existing
canonical interview hash. The page query, credentials and previous hash are
excluded. No account identifier, post body, saved records, learning evidence or
draft is serialized. Saved experiences and learning-state filters are interpreted
using the opener's own workspace; that explanation accompanies personal-scope
copy feedback and manual fallback.

Copying is an explicit native clipboard action. It does not navigate, publish,
change filters, write storage or call the API. The button keeps its label and
keyboard focus while busy, uses an immediate duplicate-command guard and reports
progress through a live status. Clipboard denial, missing support or an eight-second
wait yields an exact read-only manual link; focusing the field selects its text.
The fallback does not take focus from another task. Unsupported page protocols
show an error rather than retaining an earlier manual URL.

An owner-and-route keyed child isolates each copy attempt. Late acknowledgements
cannot mark a newer filter, returning scope or different account as copied.
Unmounting clears its deadline. Already-started native clipboard writes remain
managed by the browser; these guards control the page's feedback.

The feature stays with the deferred interview page and uses existing route and
sign-in continuation formats. Controls use 14px labels, at least 40px height on
desktop and 44px on coarse pointers, with 16px manual-link text, visible focus and
bounded wrapping.

The focused check covers URL round trips, Unicode, login continuation, native
button focus, duplicate commands, locale/field continuity, manual fallback,
timeout/retry and route/account/ABA/unmount races. The actual hub/composer check
also verifies applied versus unsubmitted search, no API/navigation/publication,
exact working draft, editor identity, selection/direction and both scroll axes.
Complete frontend checks, edition builds, actual publication and live assets are
recorded separately. Clipboard permission UI and browser visual layout are not
claimed as manually tested.
