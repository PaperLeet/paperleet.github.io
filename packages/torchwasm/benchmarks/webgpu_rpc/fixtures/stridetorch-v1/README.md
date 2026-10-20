# StrideTorch v1 compatibility fixture

Exact Apache-2.0 sources from commit
`75c0543dbd5d3db783f0bf37b6a74929daee0878`, before the TorchWasm rename.
These files retain the original advertised matrix protocol name so tests catch
accidental wire incompatibility when renaming the distribution.

- `rpc-client.mjs`: `821155085f2566f8a8376357143dccbf51277da955d0edf51dc22117a631527d`
- `rpc-protocol.mjs`: `e2c27e380aae4e41ff0ece2e3adb1059ac785d2ca8c39c76fcf7840974829d92`
