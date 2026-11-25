# VRAM budget: fitting three models in 8 GB

Measured in Phase 0 with `scripts/measure_vram.py`; raw data in
`eval/results/phase0_vram.json`.

**Setup:** RTX 5060 8 GB (8151 MiB) on Windows 11, Ollama 0.40.16 (CUDA backend),
qwen3:8b (Q4_K_M), bge-m3 and bge-reranker-v2-m3 in fp16.
**Test question:** the reranker scores 30 passages of about 450 tokens; the LLM
reads 5 of them (about 2,400 prompt tokens) and writes 239 tokens.

## What each component takes on the GPU

| Component | Memory | How we know |
|---|---|---|
| qwen3:8b weights | 4,644 MiB (+334 MiB kept in RAM) | Ollama log |
| LLM KV cache, 4k context | 576 MiB (fp16) / 306 MiB (q8_0) | Ollama log; 144 KiB per token in fp16 |
| LLM KV cache, 8k context | 1,152 MiB (fp16) / 612 MiB (q8_0) | Ollama log |
| LLM compute buffer | about 110 MiB | Ollama log |
| bge-m3 embedder | about 1,255 MiB (peak about 1.5 GB while ingesting a batch of 16) | nvidia-smi |
| bge-reranker-v2-m3 | about 1,630 MiB after scoring 30 passages | nvidia-smi |
| CUDA context of our Python process | about 160 MiB | difference between runs |
| Windows + desktop | 510 to 830 MiB with apps closed, 1.0 to 2.0 GB with apps open | nvidia-smi |

Everything on the GPU at 4k context needs about 9.3 GB with apps open, about
8.7 GB with apps closed. The card has 8.0 GB, and in practice about 7.5 GB is
usable before Windows starts paging (see lesson 1).

## Which configurations work

Time per question = embed the question + rerank 30 passages + generate the answer.

| Embedder | Reranker | Context | Apps open | Apps closed | Apps closed + q8_0 KV cache |
|---|---|---|---|---|---|
| GPU | GPU | 8k | paging, 23.0 s | paging, 19.7 s | paging, 22.6 s |
| GPU | GPU | 4k | paging, 20.1 s | OK, 6.2 s | partial paging, 8.6 s |
| CPU | GPU | 8k | paging, 22.8 s | partial paging, 7.3 s | paging, 18.9 s |
| **CPU** | **GPU** | **4k** | **OK, 6.9 s** | **OK, 5.8 s** | **OK, 5.1 s** |
| CPU | CPU | 8k | LLM alone, 60 s (rerank on CPU: 54 s) | 57 s | 55 s |

Speeds of the chosen configuration (CPU / GPU / 4k): generation 59 to 70 tokens/s,
prompt reading 2,500 to 2,950 tokens/s, 500 to 650 MiB of VRAM still free.
When paging, prompt reading drops to about 300 tokens/s (9 times slower) and
generation to about 25 tokens/s.

## Lessons

1. **Windows does not fail when VRAM is over-full: it pages.** The driver moves
   GPU memory to normal RAM. Everything still works, and Ollama still reports
   "100% GPU" and "37/37 layers on GPU", but it is 3 to 9 times slower. The
   only reliable symptom is speed, so the script compares speeds.
2. **Ollama cannot see our models' memory.** At every load the Ollama log says
   "7031 MiB free", even when the reranker already holds about 1.8 GB. So
   Ollama never moves layers to the CPU by itself here: we must budget the
   memory ourselves, through the settings.
3. **The limit is sharp.** Runs that peaked above about 7.6 GB paged, and runs
   that left 500 MiB or more free did not. Everything on the GPU at 4k is right
   at the edge: it passed in one run and partly paged in another.
4. **The reranker needs the GPU:** 0.7 to 1.0 s for 30 passages on the GPU,
   50 to 54 s on the CPU (8-core Ryzen 7 5700X).
5. **The embedder does not need the GPU at question time.** One question takes
   0.25 to 0.35 s on the CPU. Ingestion is different: on the CPU it takes about
   1.8 s per chunk, versus about 0.05 s on the GPU. So ingestion runs on the GPU,
   in its own process, while the LLM is unloaded.
6. **The q8_0 KV cache stores the LLM's attention cache in 8 bits instead of 16.**
   It saves 270 MiB at 4k and 540 MiB at 8k, and generation is about 6% slower
   (66 versus 70 tokens/s). That is not enough to make 8k fit next to the
   reranker: the 4.6 GB of LLM weights dominate.

## Chosen configuration

| Where | Embedder | Reranker | LLM |
|---|---|---|---|
| Questions (backend) | CPU | GPU, fp16 | qwen3:8b on GPU, context 4096 |
| Ingestion (CLI) | GPU, fp16 | not used | unloaded |

- Ollama settings: `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`.
  This leaves 270 MiB of extra room for the screen recorder during the demo.
- A 4k context is enough for the default: about 250 tokens of instructions,
  5 passages of about 450 tokens, the question, and an answer of up to about
  700 tokens, about 3.3k tokens in total.
- Experiments that need a longer context (800-token chunks) will run retrieval
  and generation as separate steps, so that the LLM has the GPU to itself. The
  LLM alone at 8k leaves 1.5 GB free.
