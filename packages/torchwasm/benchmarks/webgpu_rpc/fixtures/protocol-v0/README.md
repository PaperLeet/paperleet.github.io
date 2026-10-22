# Unadvertised protocol fixture

These are byte-identical snapshots of the pre-input-retention developer GPU
service, planner and client qualified in experiment 538. They intentionally omit
the new capability advertisement and cannot retain uploaded program inputs.
They are test-only inputs for cross-version MessageChannel compatibility.
Do not use them as a production fallback or modify them to satisfy a new test.

- `gpu-service.mjs`: SHA-256 `ddb5a3c6305e7efdbbc82b174804d8de429b53d761ab1d3682ed27522d8fe54b`
- `rpc-client.mjs`: SHA-256 `d957e947748db6433535c2d26a384b3504fc549b9f05f5a7a6002b6e5bf80d38`
- `plan-program.mjs`: SHA-256 `48f840e4f2b1ee53b62923546f9748a96fbcb5bd9db3e1d7db9b9e169365f0c1`
