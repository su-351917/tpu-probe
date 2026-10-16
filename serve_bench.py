# serve_bench.py : batch-serving throughput of several open models on TPU v5e-8 with vLLM (offline batch, many concurrent users).
# For each model: load (tp = 1 for small dense, 8 for MoE), generate for N prompts at several max_num_seqs, report prefill/decode tok/s and cost/1M tokens.
import os, time, json, shutil, sys, traceback
MODELS = os.environ.get("MODELS", "LiquidAI/LFM2.5-350M:1,Qwen/Qwen3.5-0.8B:1,LiquidAI/LFM2.5-1.2B-JP-202606:1,LiquidAI/LFM2.5-8B-A1B:8,LiquidAI/LFM2.5-24B-A2B:8,Qwen/Qwen3.5-35B-A3B:8").split(",")
N, PLEN, OUT = int(os.environ.get("N", "256")), int(os.environ.get("PLEN", "512")), int(os.environ.get("OUT", "256"))
SEQS = [int(x) for x in os.environ.get("SEQS", "16,64,256").split(",")]
PRICE_PER_CHIP_H = 1.20   # Cloud TPU v5e on-demand, USD per chip-hour (list price), for cost/1M-token estimates
para = ("東京メトロの南北線から山手線に乗り換えられる駅はありますか。西日暮里まで行きたいです。後楽園から乗って一番近い駅で乗り換えたいので、どの駅で降りて内回りと外回りどちらに乗ったらいいか教えてください。"
        "The committee reviewed the quarterly report and noted several items for follow-up before the next meeting. ")
res = []
for spec in MODELS:
    name, tp = spec.split(":"); tp = int(tp); t0 = time.time()
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        ids = tok(para * 20, add_special_tokens=False).input_ids[:PLEN]; prompt = tok.decode(ids)
        prompts = [f"[{i}] " + prompt for i in range(N)]
        llm = LLM(model=name, tensor_parallel_size=tp, max_model_len=PLEN + OUT + 64, max_num_seqs=max(SEQS), dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.9, enforce_eager=False)
        load_s = time.time() - t0
        # prefill-only pass (1 output token) and full pass, per concurrency level
        for ms in SEQS:
            sp1 = SamplingParams(max_tokens=1, temperature=0); sp = SamplingParams(max_tokens=OUT, temperature=0, ignore_eos=True)
            sub = prompts[:min(N, ms * 4)]
            t1 = time.time(); llm.generate(sub, sp1, use_tqdm=False); t_pre = time.time() - t1
            t2 = time.time(); outs = llm.generate(sub, sp, use_tqdm=False); t_full = time.time() - t2
            ptoks = sum(len(o.prompt_token_ids) for o in outs); otoks = sum(len(o.outputs[0].token_ids) for o in outs)
            dec_s = max(t_full - t_pre, 1e-6); chips = tp
            r = {"model": name, "tp": tp, "max_num_seqs": ms, "n_prompts": len(sub), "prompt_tok": ptoks, "out_tok": otoks, "prefill_s": round(t_pre, 2), "total_s": round(t_full, 2),
                 "prefill_tok_s": round(ptoks / t_pre), "decode_tok_s": round(otoks / dec_s), "decode_tok_s_per_chip": round(otoks / dec_s / chips),
                 "usd_per_1M_out_tok": round(PRICE_PER_CHIP_H * chips / 3600 / (otoks / dec_s) * 1e6, 3), "load_s": round(load_s)}
            print("SERVE_JSON", json.dumps(r, ensure_ascii=False), flush=True); res.append(r)
        del llm
    except Exception as e:
        print("SERVE_FAIL", name, repr(e)[:400], flush=True); traceback.print_exc()
    finally:
        try: shutil.rmtree(os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")) + "/hub", ignore_errors=True)   # free disk for the next model
        except Exception: pass
open("result_serve_bench.jsonl", "a").write("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n")
print("SERVE_DONE", flush=True)
