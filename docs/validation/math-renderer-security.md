# Math renderer security and compatibility

KaTeX is fixed at 0.18.2, the first release that addresses
[GHSA-238p-pmpm-9mq7](https://github.com/advisories/GHSA-238p-pmpm-9mq7).
The npm override keeps the direct renderer, `rehype-katex` and
`micromark-extension-math` on the same patched implementation. Updating only
the direct dependency leaves the plugins' `^0.16.0` dependency paths vulnerable.
The existing plugin majors and Commander version remain unchanged.

The advisory requires existing prototype pollution or attacker control over
an options object's prototype; KaTeX does not create that prerequisite. A
controlled fixture reproduces inherited settings enabling trusted math HTML
and inherited namespace entries becoming macros. This does not establish an
application endpoint that causes prototype pollution.

`node scripts/check-math-security.mjs` checks both community and reading
renderers, plus the KaTeX implementation resolved by every consumer. Inherited
`trust`, `default`, `processor` and macro entries must not change trust policy.
Explicit own-property options still work. Tests restore every prototype
descriptor and render static markup without opening sockets or fetching images.

Compatibility checks retain actual inline/display math, fractions, aligned
equations and matrices. KaTeX 0.18 prefixes internal CSS classes; PaperLeet's
custom math rules target the stable outer `.katex-display` class and the
version-matched KaTeX stylesheet remains imported. Existing community tests
also cover code literals, invalid formulas, macro limits/isolation, GFM and
blocked external resources. Run them and the production build when changing
the pin or math plugins; do not replace the override with an audit force fix.
