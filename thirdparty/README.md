# Third-party benchmark sources

BFCL and AgentLongBench are Git submodules pinned to specific upstream commits. Initialize them after cloning DLite:

```bash
git submodule update --init --recursive
```

Or initialize them while cloning:

```bash
git clone --recurse-submodules https://github.com/SJTU-DAI-Sys/DLite.git
```

Git checks out the revisions recorded by DLite:

- **BFCL:** [source and tool definitions](BFCL_v4_memory_source/berkeley-function-call-leaderboard/), revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. The harness uses `multi_turn_long_context`.
- **AgentLongBench:** [source](AgentLongBench/), revision `57e13e82ba20867fff3392903db2e52cf08bcc05`.

Then run `python -m dlite.agentic prepare`. It reuses existing raw data and downloads missing data; use `--download never` for local-only preparation. AgentLongBench's dataset is downloaded separately from its source checkout. Alternative source and data locations can be supplied with `--source-root` and `--data-root`.

See the [agentic evaluation guide](../dlite/agentic/README.md) for the measurement protocol and launch commands. 

Retain upstream licenses and attribution.