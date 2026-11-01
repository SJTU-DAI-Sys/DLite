"""Single-request OpenAI-compatible text server for DLite agent evaluation.

Nothing loads a model until main() starts the service. Run one process per GPU.
The benchmark harness owns tools, memory and retrieval; this entry does not score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import uuid

from .common import fingerprint, parse_config


CONTEXT_HEADERS = {
    "x-agent-benchmark": "benchmark", "x-agent-case-id": "case_id",
    "x-agent-phase": "phase", "x-agent-model-role": "model_role",
    "x-agent-turn": "turn", "x-agent-step": "step",
    "x-agent-run-id": "client_run_id",
}


class InvalidRequest(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidRequest(message)


def positive_int(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{name} must be a positive integer")
    return value


def load_rope_config(path, mode, args):
    """Explicit Qwen3 load-time override; never silently extend other targets."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path)
    native_length = int(config.get_text_config().max_position_embeddings)
    if mode != "config":
        require(config.model_type == "qwen3", "RoPE overrides here support Qwen3 only")
        require(args.context_limit is not None, "RoPE override requires explicit context-limit")
        existing = getattr(config, "rope_parameters", None) or {}
        theta = existing.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        scaling = ({"rope_type": "yarn", "factor": args.rope_factor,
                    "original_max_position_embeddings": args.original_context}
                   if mode == "yarn" else None)
        config.rope_scaling = scaling
        config.rope_parameters = {"rope_theta": theta, **(scaling or {"rope_type": "default"})}
        config.max_position_embeddings = args.context_limit
    return config, native_length


def plain_messages(messages):
    require(isinstance(messages, list) and messages, "messages must be a nonempty list")
    result = []
    for message in messages:
        require(isinstance(message, dict), "each message must be an object")
        require(message.get("role") in {"system", "user", "assistant", "tool"},
                "unsupported message role")
        message = dict(message)
        content = message.get("content")
        if content is None and message.get("tool_calls"):
            content = ""
        if isinstance(content, list):
            parts = []
            for part in content:
                require(isinstance(part, dict) and part.get("type") == "text"
                        and isinstance(part.get("text"), str),
                        "DLite is text-only: image/audio/video content is unsupported; "
                        "use a multimodal endpoint or an explicitly labelled text variant")
                parts.append(part["text"])
            content = "\n".join(parts)
        require(isinstance(content, str), "message content must be text")
        message["content"] = content
        result.append(message)
    return result


class Runtime:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.run_id = args.run_id or uuid.uuid4().hex
        self.metrics = Path(args.metrics).resolve()
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        # Existing logs are never appended across service lifetimes.
        self.log = self.metrics.open("x", encoding="utf-8", buffering=1)
        import torch
        import transformers
        from dlite.engine import DLiteEngine
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        target_config, source_native = load_rope_config(
            args.target_model, args.target_rope_scaling, args)
        if args.backend == "sd":
            if args.target_rope_scaling == "config" and args.draft_rope_scaling == "config":
                self.engine = DLiteEngine.from_pretrained(
                    args.target_model, args.draft_model, device=args.device,
                    attention_backend=args.attention_backend)
            else:
                from dlite.modeling import DLiteDraftModel
                draft_config, _ = load_rope_config(args.draft_model, args.draft_rope_scaling, args)
                common = {"dtype": torch.bfloat16, "attn_implementation": args.attention_backend}
                target = AutoModelForCausalLM.from_pretrained(
                    args.target_model, config=target_config, **common).to(args.device).eval()
                draft = DLiteDraftModel.from_pretrained(
                    args.draft_model, config=draft_config, **common).to(args.device).eval()
                tokenizer = AutoTokenizer.from_pretrained(args.target_model)
                self.engine = DLiteEngine(target, draft, tokenizer)
            self.target, self.tokenizer = self.engine.target, self.engine.tokenizer
            maximum = self.engine.draft.proposal_length + 1
            self.block_size = maximum if args.verify_block_size is None else args.verify_block_size
            require(1 <= self.block_size <= maximum, "verify block size exceeds draft capacity")
        else:
            self.engine = None
            self.target = AutoModelForCausalLM.from_pretrained(
                args.target_model, config=target_config, dtype=torch.bfloat16,
                attn_implementation=args.attention_backend).to(args.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(args.target_model)
            self.block_size = None
        config = self.target.config.get_text_config()
        native = int(config.max_position_embeddings)
        self.context_limit = native if args.context_limit is None else args.context_limit
        require(0 < self.context_limit <= native,
                "context limit must fit the loaded target config; configure RoPE separately")
        if args.compact_prefill:
            require(args.backend == "sd" and config.model_type == "qwen3",
                    "compact prefill requires the frozen Qwen3 SD runtime")
            from compact_prefill import install_compact_prefill
            install_compact_prefill(self.target, self.engine.draft.target_layer_ids,
                                    self.engine.draft.config.num_target_layers)
        self.write({"type": "service_start", "backend": args.backend,
                    "target_model": args.target_model, "draft_model": args.draft_model,
                    "served_model": args.served_model, "runtime_root": args.runtime_root,
                    "attention_backend": args.attention_backend,
                    "verify_block_size": self.block_size, "context_limit": self.context_limit,
                    "max_output_tokens": args.max_output_tokens,
                    "preallocate_kv_cache": args.preallocate_kv_cache,
                    "compile_sequential_head": args.compile_sequential_head,
                    "compact_prefill": args.compact_prefill,
                    "source_native_context": source_native,
                    "target_rope_scaling": args.target_rope_scaling,
                    "draft_rope_scaling": args.draft_rope_scaling if self.engine else None,
                    "target_rope_parameters": getattr(config, "rope_parameters", None),
                    "draft_rope_parameters": getattr(self.engine.draft.config, "rope_parameters", None)
                                             if self.engine else None,
                    "allow_ar_baseline": args.allow_ar_baseline,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "source_sha256": fingerprint([
                        Path(args.runtime_root) / "dlite/engine.py",
                        Path(args.runtime_root) / "dlite/modeling/dlite.py",
                        Path(args.runtime_root) / "compact_prefill.py",
                        Path(args.target_model) / "config.json",
                        *([Path(args.draft_model) / "config.json"] if args.draft_model else []),
                        Path(__file__)]),
                    "protocol": "greedy text; decode excludes prefill; no task scoring"})

    def write(self, row):
        row = {"run_id": self.run_id, "timestamp": time.time(), **row}
        with self.log_lock:
            self.log.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sync_time(self):
        if self.target.device.type == "cuda":
            self.torch.cuda.synchronize(self.target.device)
        return time.perf_counter()

    def new_cache(self, length):
        if self.args.preallocate_kv_cache:
            from dlite.modeling.kv_cache import make_target_cache
            return make_target_cache(self.target, length)
        from transformers import DynamicCache
        return DynamicCache(config=self.target.config)

    def autoregressive(self, ids, max_tokens, stop_ids):
        torch = self.torch
        cache = self.new_cache(ids.shape[1] + max_tokens)
        output = self.target(ids, past_key_values=cache, use_cache=True,
                             logits_to_keep=1, output_hidden_states=False)
        token = output.logits[:, -1:].argmax(-1)
        del output
        started = self.sync_time()
        generated = []
        for _ in range(max_tokens):
            generated.append(token)
            if int(token.item()) in stop_ids or len(generated) == max_tokens:
                break
            output = self.target(token, past_key_values=cache, use_cache=True,
                                 logits_to_keep=1, output_hidden_states=False)
            token = output.logits[:, -1:].argmax(-1)
            del output
        elapsed = self.sync_time() - started
        return torch.cat(generated, dim=1), {"accept_lengths": [], "decode_wall_time": elapsed}

    def validate(self, body, chat):
        require(isinstance(body, dict), "JSON body must be an object")
        require(body.get("model") in {self.args.served_model, self.args.target_model},
                "model must match the service's served model or target path")
        require(body.get("stream", False) is False, "streaming is unsupported")
        require(body.get("n", 1) == 1 and body.get("best_of", 1) == 1,
                "only n=1 and best_of=1 are supported")
        require(body.get("temperature", 0) == 0, "this integration uses temperature=0")
        for key, allowed in {"top_p": (1, None), "top_k": (-1, 0, None),
                             "presence_penalty": (0, None), "frequency_penalty": (0, None),
                             "repetition_penalty": (1, None), "logprobs": (None, False),
                             "echo": (False,), "suffix": (None,),
                             "response_format": (None,), "seed": (None, 0)}.items():
            if key in body:
                require(body[key] in allowed, f"unsupported {key}: {body[key]!r}")
        for key in ("logit_bias", "guided_json", "json_schema", "grammar", "reasoning_effort"):
            require(not body.get(key), f"unsupported parameter: {key}")
        require(not body.get("tools") and not body.get("functions"),
                "native tools are unsupported here; BFCL owns its tool prompt and parser")
        require(body.get("tool_choice") in (None, "none"), "unsupported tool_choice")
        limits = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
        for limit in limits:
            positive_int(limit, "max_tokens/max_completion_tokens")
        require(len(set(limits)) <= 1, "conflicting token limits")
        max_tokens = limits[0] if limits else self.args.max_output_tokens
        require(max_tokens <= self.args.max_output_tokens,
                "requested output exceeds service max-output-tokens; raise the service cap explicitly")
        template = body.get("chat_template_kwargs") or {}
        require(isinstance(template, dict) and set(template) <= {"enable_thinking"},
                "unsupported chat template parameters")
        require(isinstance(template.get("enable_thinking", self.args.enable_thinking), bool),
                "enable_thinking must be boolean")
        if chat:
            messages = plain_messages(body.get("messages"))
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=template.get("enable_thinking", self.args.enable_thinking))
        else:
            prompt = body.get("prompt")
            require(isinstance(prompt, str) and prompt, "prompt must be a nonempty raw string")
            require(not template, "raw completions already have a template; do not pass template kwargs")
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        require(isinstance(stops, list) and all(isinstance(s, str) and s for s in stops),
                "stop must be a string or list of nonempty strings")
        stop_ids = body.get("stop_token_ids") or []
        require(isinstance(stop_ids, list) and all(isinstance(i, int) and not isinstance(i, bool)
                and 0 <= i < len(self.tokenizer) for i in stop_ids), "invalid stop_token_ids")
        stop_ids = set(stop_ids)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            stop_ids.add(eos)
        for stop in stops:
            tokens = self.tokenizer.encode(stop, add_special_tokens=False)
            if len(tokens) == 1:
                stop_ids.add(tokens[0])
        require(isinstance(body.get("skip_special_tokens", True), bool),
                "skip_special_tokens must be boolean")
        return prompt, max_tokens, stops, stop_ids

    def handle(self, body, chat, context, arrived):
        request_id = "chatcmpl-" + uuid.uuid4().hex if chat else "cmpl-" + uuid.uuid4().hex
        row = {"type": "request", "request_id": request_id, "backend": self.args.backend,
               "target_model": self.args.target_model, "draft_model": self.args.draft_model,
               "endpoint": "chat" if chat else "completions", **context}
        with self.lock:
            started = time.perf_counter()
            row["queue_seconds"] = started - arrived
            try:
                require(isinstance(body, dict), "JSON body must be an object")
                active_backend = body.get("decode_backend", self.args.backend)
                require(active_backend == self.args.backend,
                        "AR and SD require separate service lifetimes and complete task runs")
                row["backend"] = active_backend
                prompt, max_tokens, stops, stop_ids = self.validate(body, chat)
                ids = self.tokenizer(prompt, add_special_tokens=False,
                                     return_tensors="pt").input_ids.to(self.target.device)
                num_input = ids.shape[1]
                reserve = (self.block_size or 0) if active_backend == "sd" else 0
                require(num_input + max_tokens + reserve <= self.context_limit,
                        f"context capacity exceeded: input={num_input}, output={max_tokens}, "
                        f"verification reserve={reserve}, limit={self.context_limit}; no truncation")
                row.update(input_tokens=num_input, requested_max_tokens=max_tokens,
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                with self.torch.inference_mode():
                    generation_start = self.sync_time()
                    if active_backend == "sd":
                        cache = (self.new_cache(num_input + max_tokens + reserve)
                                 if self.args.preallocate_kv_cache else None)
                        output = self.engine.draft.spec_generate(
                            target=self.target, input_ids=ids, max_new_tokens=max_tokens,
                            stop_token_ids=sorted(stop_ids), temperature=0.0,
                            verify_block_size=self.block_size, stochastic_verification_mode="match",
                            compile_serial_head=self.args.compile_sequential_head,
                            target_cache=cache)
                        generated = output[:, num_input:]
                        stats = dict(self.engine.draft.get_last_decode_stats())
                        stats["accept_lengths"] = list(stats.get("accept_lengths", []))
                    else:
                        generated, stats = self.autoregressive(ids, max_tokens, stop_ids)
                    generation_seconds = self.sync_time() - generation_start
                token_ids = generated[0].tolist()
                text = self.tokenizer.decode(token_ids,
                    skip_special_tokens=body.get("skip_special_tokens", True))
                stop_positions = [text.index(stop) for stop in stops if stop in text]
                stopped = any(token in stop_ids for token in token_ids)
                if stop_positions:
                    text = text[:min(stop_positions)]
                    stopped = True
                lengths = stats["accept_lengths"]
                steps = len(lengths)
                accepted = sum(length - 1 for length in lengths)
                row.update(status="ok", output_tokens=len(token_ids),
                           decode_output_tokens_excluding_prefill_token=max(len(token_ids) - 1, 0),
                           output_tokens_sha256=hashlib.sha256(b"".join(
                               int(token).to_bytes(4, "little") for token in token_ids)).hexdigest(),
                           visible_text_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                           verify_block_size=self.block_size if active_backend == "sd" else None,
                           accept_lengths=lengths,
                           accepted_draft_tokens=accepted,
                           verified_draft_tokens=steps * ((self.block_size or 1) - 1),
                           verification_steps=steps, decode_seconds=float(stats["decode_wall_time"]),
                           generation_seconds=generation_seconds,
                           string_stop_postprocessed=bool(stops),
                           finish_reason="stop" if stopped else "length")
                usage = {"prompt_tokens": num_input, "completion_tokens": len(token_ids),
                         "total_tokens": num_input + len(token_ids)}
                choice = {"index": 0, "finish_reason": row["finish_reason"]}
                if chat:
                    # This adapter intentionally leaves tool parsing to the harness.
                    choice["message"] = {"role": "assistant", "content": text}
                else:
                    choice.update(text=text, logprobs=None)
                result = {"id": request_id, "object": "chat.completion" if chat else "text_completion",
                          "created": int(time.time()), "model": self.args.served_model,
                          "choices": [choice], "usage": usage}
                if body.get("return_decode_metrics") is True:
                    result["dlite_metrics"] = {key: row[key] for key in (
                        "backend", "input_tokens", "output_tokens", "prompt_sha256",
                        "output_tokens_sha256", "decode_output_tokens_excluding_prefill_token",
                        "verify_block_size", "accept_lengths", "decode_seconds", "generation_seconds")}
                return result, 200, request_id
            except Exception as exc:
                bad_request = isinstance(exc, InvalidRequest)
                row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                if self.target.device.type == "cuda" and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                    self.torch.cuda.empty_cache()
                return {"error": {"message": str(exc), "type": "invalid_request_error"
                                  if bad_request else "server_error", "code": type(exc).__name__}}, \
                    400 if bad_request else 500, request_id
            finally:
                row["service_seconds"] = time.perf_counter() - started
                row["request_seconds"] = time.perf_counter() - arrived
                self.write(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--backend", choices=["ar", "sd"], required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--output", type=Path, help="Service log directory when --metrics is omitted")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runtime-root", type=str)
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--verify-block-size", type=int)
    parser.add_argument("--attention-backend", choices=["flash_attention_2", "sdpa", "eager"],
                        default="flash_attention_2")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--compile-sequential-head", action="store_true")
    parser.add_argument("--preallocate-kv-cache", action="store_true")
    parser.add_argument("--compact-prefill", action="store_true")
    parser.add_argument("--allow-ar-baseline", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--target-rope-scaling", choices=["config", "none", "yarn"], default="config")
    parser.add_argument("--draft-rope-scaling", choices=["config", "none", "yarn"], default="config")
    parser.add_argument("--rope-factor", type=float, default=4.0)
    parser.add_argument("--original-context", type=int, default=40960)
    args = parse_config(parser, argv)
    if args.allow_ar_baseline:
        parser.error("paired replay is disabled; start an independent AR service and run")
    if args.backend == "sd" and not args.draft_model:
        parser.error("--draft-model is required for SD")
    if args.max_output_tokens < 1 or not 0 < args.port < 65536:
        parser.error("invalid output limit or port")
    if args.rope_factor < 1 or args.original_context < 1:
        parser.error("invalid RoPE factor or original context")
    if args.metrics is None:
        if args.output is None:
            parser.error("set --metrics or --output for this service lifetime")
        args.metrics = str(args.output / "server_metrics.jsonl")
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    import fcntl
    import os
    import subprocess
    import tempfile
    guard = open(Path(tempfile.gettempdir()) / f"dlite-agent-{os.getuid()}-gpu{args.gpu}.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = int(subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"
    ], text=True).strip())
    if used > 1024:
        raise RuntimeError(f"GPU {args.gpu} is occupied ({used} MiB); select a free device")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    root = Path(args.runtime_root) if args.runtime_root else args.project_root
    sys.path.insert(0, str(root.resolve()))
    if root.resolve() != args.project_root.resolve():
        import dlite
        runtime_package = root.resolve() / "dlite"
        if not runtime_package.is_dir():
            parser.error("runtime-root must contain a dlite package")
        if "dlite.engine" in sys.modules or "dlite.modeling" in sys.modules:
            parser.error("select runtime-root before loading the inference API")
        dlite.__path__.insert(0, str(runtime_package))
    args.runtime_root = str(root.resolve())
    # HTTP and ML dependencies are only required after parsing an explicit serve command.
    global Request
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    runtime = Runtime(args)
    app = FastAPI(title="DLite agent evaluation", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": args.served_model, "object": "model",
                                              "owned_by": "local"}]}

    @app.get("/health")
    def health():
        return {"status": "ready", "backend": args.backend, "run_id": runtime.run_id,
                "served_model": args.served_model, "target_model": args.target_model,
                "context_limit": runtime.context_limit, "max_output_tokens": args.max_output_tokens,
                "verify_block_size": runtime.block_size, "allow_ar_baseline": args.allow_ar_baseline,
                "target_rope_scaling": args.target_rope_scaling,
                "draft_rope_scaling": args.draft_rope_scaling if runtime.engine else None,
                "rope_factor": args.rope_factor, "original_context": args.original_context,
                "attention_backend": args.attention_backend,
                "compile_sequential_head": args.compile_sequential_head,
                "preallocate_kv_cache": args.preallocate_kv_cache,
                "runtime_root": args.runtime_root}

    async def completion(request, chat):
        arrived = time.perf_counter()
        context = {name: request.headers[header] for header, name in CONTEXT_HEADERS.items()
                   if header in request.headers}
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse({"error": {"message": "invalid JSON body",
                                            "type": "invalid_request_error"}}, status_code=400)
        result, status, request_id = await run_in_threadpool(
            runtime.handle, body, chat, context, arrived)
        return JSONResponse(result, status_code=status, headers={"x-request-id": request_id})

    @app.post("/v1/completions")
    async def text_completion(request: Request):
        return await completion(request, False)

    @app.post("/v1/chat/completions")
    async def chat_completion(request: Request):
        return await completion(request, True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
