# Deferred paper details

The homepage previously loaded the complete paper-detail renderer alongside the
searchable library. Details now load as an independent route through the existing
deferred-feature boundary. The renderer body is unchanged; shared state icons and
progress bars retain their existing markup. No judge, catalog or persistence
contract changes.

Compared with main `d968be6e`, identical edition configurations produce these
complete initial static JavaScript graphs, including every static dependency:

| Edition | Raw before | Raw after | Gzip before | Gzip after |
| --- | ---: | ---: | ---: | ---: |
| Hosted | 409,168 | 402,739 | 132,366 | 130,615 |
| Connected | 403,654 | 397,227 | 130,723 | 128,989 |
| Offline | 401,842 | 395,416 | 130,095 | 128,400 |
| Actual Pages configuration | 409,455 | 403,026 | 132,503 | 130,758 |

Each initial graph still contains three JavaScript assets. The Pages profile uses
the hosted edition, the configured API origin and enabled account-app links. Its
raw budget margin grows from 145 to 6,574 bytes. The independent paper-detail
chunk is 7,516 raw bytes in the offline build and is excluded from the initial
graph; a direct paper link now requests this route on demand. These are build-byte
measurements, not browser latency or CPU benchmarks. Existing raw/gzip caps remain.

A real browser test temporarily removed the built detail script and then restored
its exact bytes. Ordinary repeated import remained failed in that document. A
fresh document recovered. The route therefore offers an explicit reload using
the App's existing workspace/run/save/conflict guard, plus a return link retaining
the current library filters. Pending saves disable recovery. It does not reload
automatically. The fresh-document helper preserves the current route and retires
its temporary cache-refresh parameter after startup.

Complete frontend checks and hosted/connected/offline/Pages builds pass. Mounted
Chinese and English checks cover pending imports, isolated errors, guarded reload,
latest selection/return props, full curriculum, unchanged learner storage, and
empty/loading progress. The build check requires a separate dynamic detail entry.
Actual browser checks verify deep links, search returns, the scoped module-29
recommendation, restored language preference and recovery after the missing-script
fault. A preexisting 955-byte UTF-8 learner draft remains byte-identical after
recovery. No account or formal submission records are created.
