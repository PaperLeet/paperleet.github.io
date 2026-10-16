# Readable diagnostic and code-navigation controls

The main reading scale was larger, but the new Python definition picker still
used 30px controls. Error navigation also used 30px arrows, while full tracebacks,
copy feedback and several repair actions retained smaller text. These gaps made
the inspect → edit → rerun path feel inconsistent with the surrounding workspace.

This stylesheet-only refinement gives definition selection/activation, failure
selection/arrows, code-location actions, copy, result filters, disclosure actions
and detail retries at least 40px desktop targets and 44px coarse-pointer targets.
Native selects use the 16px body scale on coarse pointers; ordinary labels and
actions use the shared 14px scale. Diagnostics, code context, source notices and
property explanations use 14px, with the existing 16px workspace summary rule
retained. Dates and compact layer/duration metadata keep a 13px hierarchy.

Failure arrows and the picker share the same height. The source-context gutter
uses a five-character width and taller rows. Workflow steps, result filters,
diagnostic toolbars and action labels can wrap when space or preferred text size
requires it. Labels and native controls retain their existing behavior. Focus
rings on failure navigation and native result/trace disclosures sit inside their
boxes, so existing overflow clipping does not hide the outline.

The scope is three stylesheets and this document. No editor source, cursor,
diagnostic ownership, account return, report identity, submission, execution,
verdict, retry or persistence code changes. No dependency or runtime changes are
required. Existing mounted checks cover the actual CodeMirror picker, error
selection and line navigation, current-source edit/retry guards, disclosure/copy
behavior, keyboard ownership and retained draft/selection/scroll. Source and
compiled-CSS review establish the intended declarations; these are not browser
layout or visual tests. Complete frontend checks, both edition builds and actual
cloud/deployment/live-asset checks are recorded separately for the release.
