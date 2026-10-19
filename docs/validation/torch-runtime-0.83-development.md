# Torch runtime 0.83 development candidate

The 0.83 candidate adopts the qualified binary operand-lifetime change described
in the [owned-flow record](torch-operand-lifetime-owned-flow.md). Its numerical
source matches the frozen measured 0.82 candidate; the package version literal
changes to 0.83. No terminal-HVP or production GPU optimization is included.

Ordinary multiplication and direct division retain shape metadata and only
operand values needed by active derivatives. Existing version counters,
saved-graph consumption and numerical helpers remain unchanged. Ten new SDK
cases protect unused-storage release, broadcasting and required saved operands.
The package still has 66 runtime/license resources and NumPy as its sole runtime
dependency. The StrideTorch name is retained pending a naming decision.

The measured candidate passes the prior full SDK in two native NumPy
environments and actual Node/Chrome WASM, independent numerical controls and
the new lifetime regressions. Full-call traced allocation peaks fall about 83%
in the representative large scalar chain, while the largest predefined
three-process aggregate latency cost is about 7.9%. All timing samples and one
flagged individual Node control remain in the record. This is a scoped
memory/latency tradeoff.

Final-version SDK, distribution, PaperLeet catalog/export and browser integration
checks remain pending. Main continues to use the verified 0.82 version until
those checks pass. No CI, tag, publication or deployment is requested.
