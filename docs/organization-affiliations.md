# Organization affiliation display

Reviewed 2026-10-15. These mappings translate already-public, verified-mailbox affiliation data into readable organization names. They are display metadata, not an email registration allowlist, enrollment/employment verification, or authorization rule. Backend account verification remains authoritative.

`AffiliationBadge` shows the localized organization name and a bundled logo when available. It never displays the email address or the raw domain. Recognized domains and their subdomains map to a reviewed display name; suffix matches require a dot boundary. Unknown domains produce a readable organization label with a neutral building icon, never a guessed university name or official logo. Server-provided readable organization names are preserved.

Logo files and their official source URLs are recorded in [asset sources](../web/public/organization-logos/SOURCES.md). Images are served locally; profiles do not request avatars or logos from external services based on email domains. Missing/failed images use a fixed-size neutral icon. Company names and IDs follow the interview catalog; its official websites supply their matching public domain names. No new authenticated API is needed.

## University display sources

Chinese names are localized display labels. English names and domain ownership were checked against these official sites. `uw.edu` redirects to `www.washington.edu`; `www.ust.hk` redirects to `hkust.edu.hk`.

| Organization | Domains | Official source |
| --- | --- | --- |
| Brown University | `brown.edu` | [Website](https://www.brown.edu/) |
| Massachusetts Institute of Technology | `mit.edu` | [Website](https://www.mit.edu/) |
| University of California, Berkeley | `berkeley.edu` | [Website](https://www.berkeley.edu/) |
| Cornell University | `cornell.edu` | [Website](https://admissions.cornell.edu/) |
| Columbia University | `columbia.edu` | [Website](https://www.columbia.edu/) |
| Stanford University | `stanford.edu` | [Website](https://www.stanford.edu/about) |
| Carnegie Mellon University | `cmu.edu` | [Website](https://www.cmu.edu/) |
| Harvard University | `harvard.edu` | [Website](https://www.harvard.edu/about/) |
| University of Pennsylvania | `upenn.edu` | [Website](https://www.upenn.edu/about) |
| University of California, Los Angeles | `ucla.edu` | [Website](https://www.ucla.edu/) |
| University of California San Diego | `ucsd.edu` | [Website](https://ucsd.edu/) |
| University of Illinois Urbana-Champaign | `illinois.edu` | [Website](https://illinois.edu/) |
| University of Massachusetts Amherst | `umass.edu` | [Website](https://www.umass.edu/) |
| University of Washington | `washington.edu`, `uw.edu` | [Website](https://www.washington.edu/) |
| University of Southern California | `usc.edu` | [Website](https://www.usc.edu/) |
| University of Maryland | `umd.edu` | [Website](https://umd.edu/) |
| The University of Texas at Austin | `utexas.edu` | [Website](https://www.utexas.edu/) |
| New York University | `nyu.edu` | [Website](https://join.nyu.edu/) |
| Princeton University | `princeton.edu` | [Website](https://www.princeton.edu/) |
| Tsinghua University | `tsinghua.edu.cn` | [Website](https://www.tsinghua.edu.cn/) |
| Peking University | `pku.edu.cn` | [Website](https://www.pku.edu.cn/) |
| The University of Hong Kong | `hku.hk` | [Website](https://www.hku.hk/en/sitemap) |
| The Chinese University of Hong Kong | `cuhk.edu.hk` | [Website](https://www.cuhk.edu.hk/) |
| The Hong Kong University of Science and Technology | `hkust.edu.hk`, `ust.hk` | [Website](https://hkust.edu.hk/) |
| National University of Singapore | `nus.edu.sg` | [Website](https://nus.edu.sg/) |
| Nanyang Technological University | `ntu.edu.sg` | [Website](https://www.ntu.edu.sg/) |
