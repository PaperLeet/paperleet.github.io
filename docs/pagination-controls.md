# List pagination and input ownership

The library supports numbered pages, previous/next controls, named gap jumps, a quick page input above the list and a full pager below it. Page sizes remain 20, 50 and 100. Filters and return links retain the current scope, and changing page size retains the first visible item when possible. A page change reveals the result heading; invalid page input stays local with an associated error and Escape recovery.

Page-size and jump controls share a 44px height, tabular numbers, clear focus and hover feedback. Full pager options separate page size from direct entry; narrow containers remove that separator and wrap the fields. Reduced motion and forced colors retain visible feedback.

Jump fields check their connected active form and the parent list context before validation, error announcements, focus, Escape handling or navigation. The library, saved problems, community/profile and interview pagers supply their existing current-view checks. Locale changes alone cannot reactivate a retired list. A hidden individual pager cannot change an otherwise active list.

An actual React/JSDOM fixture reproduced a hidden invalid jump stealing focus from another control. The regression checks hidden, inert, aria-hidden and retired route cases, along with invalid-input recovery, locale-stable native input, full-width digits, composition and unmount behavior. Application-level suites also cover the real library, saved problems, community/profile and interview lists. These checks do not establish physical browser layout or operating-system focus behavior.
