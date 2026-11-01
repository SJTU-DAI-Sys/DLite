# Agentic acceptance-length evaluation

```bash
python -m dlite.agentic {prepare,serve,run,summarize}
```

## Workloads

- **BFCL V4** `multi_turn_long_context` evaluates tool use across a conversation. The native harness executes tools and includes their results in subsequent requests.
- **AgentLongBench** evaluates counting, state tracking, and reasoning over recorded agent interactions. Historical tool calls are not executed again.
- **LongMemEval V2 / AgentRunbook-R** is audit-only. Excluding entire trajectories containing images conflicts with the native screenshot requirement, so inference is unsupported under this policy.



## Measurement protocol

- Count the complete input with the target tokenizer, including templates, tool definitions, history, tool results, and the current question.
- Select **50 model requests per workload** with `32000 < input_tokens < 100000`, measured at request start. A request is a single model call, not a complete conversation.
- BFCL selects requests dynamically. Short prefixes still execute but do not enter acceptance statistics. Report fewer than 50 requests if the workload does not provide enough eligible calls.
- Run AR and SD independently with their own generated histories. AR has no acceptance-length value.
- Acceptance length includes one target anchor and is averaged over all selected verification rounds. No correctness scoring or speedup calculation is performed.



## Setup

Use Python 3.11 or later. Run commands from the repository root in the inference environment:

```bash
git submodule update --init --recursive
python -m pip install -e '.[agentic]'
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CFG="$PWD/dlite/agentic/config.example.json"
export BENCHMARK=agentlongbench  # or bfcl
```



For BFCL, install its dependencies in a separate environment:

```bash
python -m pip install -e \
  thirdparty/BFCL_v4_memory_source/berkeley-function-call-leaderboard
```

The pinned source submodules and data preparation are described in [thirdparty/README.md](../../thirdparty/README.md).

## Prepare and run

Set `PREPARED`, `SERVICE_OUTPUT`, and `RUN_OUTPUT` to fresh locations of your choice. Use the inference environment for preparation and serving:

```bash
python -m dlite.agentic prepare --config "$CFG" \
  --benchmark "$BENCHMARK" --output "$PREPARED"
```

AgentLongBench preparation selects complete rendered prompts. BFCL preparation records candidate sessions; request eligibility is determined during the native tool workflow.

Start the service in one terminal:

```bash
python -m dlite.agentic serve --config "$CFG" \
  --backend sd --output "$SERVICE_OUTPUT"
```

Run the workload in another terminal, using the separate BFCL environment for BFCL or the inference environment for AgentLongBench:

```bash
python -m dlite.agentic run --config "$CFG" \
  --benchmark "$BENCHMARK" --mode sd --prepared "$PREPARED" \
  --output "$RUN_OUTPUT"

python -m dlite.agentic summarize --run-dir "$RUN_OUTPUT"
```

For AR, restart the service with `--backend ar` and run with `--mode ar`, using fresh locations. Existing runs are not overwritten.