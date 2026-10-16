# Library topic discovery

The topic menu previously followed catalog insertion order and exposed `Top-K` and `Top-k` as separate choices. Each choice omitted the paper using the other capitalization. Topic names now appear in alphabetical/numeric order, with one option per case-insensitive name, and either spelling matches all corresponding papers.

The catalog retains its original tags. Existing filter URLs keep their exact spelling and paper order, including when entering a paper or practice module and returning. The native select displays the matching available option for a differently cased URL; it changes the URL only through the existing explicit filter action. Category, collection, difficulty, query and progress boundaries continue to use the shared library/practice selection logic. Unknown topics still retire through the existing normalization rule.

No controls, dependencies, algorithms, browser runtime bundles or private service behavior are added. The existing native control and its keyboard focus remain mounted when changing a topic.

## Verification

- The maintained `sdpo` and `llm-training-generation` papers are both found through `Top-K`, `Top-k` and mixed-case links. Their practice modules use the same results; the selected category and progress filters remain effective.
- A real mounted `PaperLibrary` selects the single topic option from an old differently cased URL. Both paper links retain that original spelling. Choosing another topic preserves select focus and the selected paper sort; clearing it preserves the sort as well.
- The existing library navigation check also covers collection/account/practice round trips, invalid destinations, bibliography search and bilingual reading controls. Source paper metadata remains byte-identical after topic discovery and filtering.
- Complete frontend checks, edition build budgets and browser preview results are recorded with the PR after execution. Release still requires the existing complete quality gate and actual publication verification.
