# Transformers benchmark

For long-context Transformers speed measurements, enable **FlashAttention, KV preallocation and Sequential-head compilation** for both the autoregressive baseline and DLite:

```bash
dlite-benchmark \
  --target-model /models/Qwen3-8B \
  --draft-model /models/dlite_qwen3_8b_full_dlitev2 \
  --dataset longbench_v2_64000_32000_single_document_qa \
  --dataset-path-file /datasets/dataset_path.json \
  --attention-backend flash_attention_2 \
  --compile-sequential-head \
  --preallocate-kv-cache \
  --temperature 0 --max-new-tokens 512 \
  --output benchmark-results/longbench.json
```

