# TP4 prefill time budget

One prompt chunk on one node of the world-4 split (node g is TP2 rank g % 2 inside pair g / 2). Pair 0 is the heavy pair, and the numbers below are that node's. Floors: 59 TFLOPS fp16 dense (WMMA) and 230 GB/s. The 230 GB/s figure is on-node DRAM (measured streaming on the 8060S); it is not the network. Roofline of a stage is max(FLOPs / 59e12, bytes / 230e9). FLOPs = 2·M·K·N. EXL3 weight bytes = K·N·k2/16 (k2 = 8 is 4.0 bpw, 0.5 B/weight; k2 = 12 is 6.0 bpw, 0.75 B/weight), plus fp16 scales suh/svh. Ridge = 59e12/230e9 = 256.5 FLOP/byte.

Every number is marked **computed** (arithmetic in the text) or **measured** (with the source). A from-start prompt does not run 40 full layers. `lanes.zig` sets replay = len − window (window = 128). `fillWhole` / `fillSplit` stop a chunk at layer 20 (`layers/2`) unless the chunk holds rows at or past replay, and the decoder layers then run on the surviving window. A chunk that ends at or before replay is encoder-only: 20 layers, no head. Counting 40 full layers overstates both the compute and the collective bytes by about 2×.

Shapes are the served config (`config.zig` `test_config`; the comment there says "EXL3 2.9 bpw mul1", the served pack is 4.0 bpw). 40 layers, D = 5120, heads = 64, head_dim = 512, q_lora = 1280, o_lora = 1024, o_groups = 8, window = 128, routed experts = 384, top_k = 6, expert width = 2304, one shared expert. `exl3_experts.zig` says a 4.0 bpw layer is routed k2 = 8 and shared k2 = 12. On 2D the shared expert is the last of the 385-expert table (`weights.Layer.shared` is null). Engram wkv k2 is not pinned in the code read for this note; bytes below assume k2 = 8, and the FLOPs do not depend on k2. The served chunk is 2,048 rows (`model.chunk_rows`), a handshake field. 8,192 tokens is four of those chunks.

## One full layer, pair 0

Attention projections, all compute-bound. At M = 2,048 the FLOP times are wq_a 0.455 ms (K = 5120, N = 1280), wq_b 0.728 (K = 1280, N = 8192), wkv 0.182 (K = 5120, N = 512), wo_a 0.582 (two groups, each K = 4096, N = 1024), wo_b 0.728 (K = 4096, N = 2560). Sum 2.675 ms. At M = 1,818 the same sum is 2.375 ms (scale by 1818/2048). **Computed.**

Routed experts, bandwidth-bound. Rows = M·6. Gate and up are two GEMMs, K = 5120, N = 640. At M = 2,048: FLOPs 1.611e11 → 2.730 ms; weight bytes 384·2·(5120·640·8/16) = 1.258e9, scales 384·2·2·(5120+640) = 8.85e6, activations 2·12288·5120·2 + 12288·640·2 = 2.674e8; total 1.535e9 bytes → 6.672 ms. Down, K = 1152, N = 2560: FLOPs 7.248e10 → 1.228 ms; weights 384·(1152·2560·0.5) = 5.662e8, scales 2.85e6, xd 2.831e7, fp32 out 1.258e8; total 7.232e8 → 3.144 ms. At M = 1,818 the byte floors are 6.541 ms (gate/up) and 3.069 ms (down). **Computed.** `pfPlan` uses nm = 64 while the routed rows·slots product is under 48·385 = 18,480, else 128. At M = 2,048 the product is 14,336, so nm = 64, about 32 rows per expert, one job, weights once. At M = 8,192 it would be nm = 128, but the served launch is still a 2,048-row chunk.

Shared expert, same column shard, k2 = 12, every row, one job per 64 rows. At M = 2,048 that is 32 jobs. Gate/up FLOPs 0.455 ms; bytes with the weight re-read 32 times 0.878 ms; weight once 0.215 ms. Down FLOPs 0.205 ms; 32× weight 0.419 ms. At M = 1,818, 29 jobs: gate/up bytes 0.792 ms, down bytes 0.378 ms. **Computed.**

Weight bytes on those projections are small next to the FLOPs. wq_a is 5120·1280·0.5 = 3.28 MB, wq_b 1280·8192·0.5 = 5.24 MB, wkv 5120·512·0.5 = 1.31 MB, wo_a 2·4096·1024·0.5 = 4.19 MB, wo_b 4096·2560·0.5 = 5.24 MB, plus a few tens of KB of fp16 scales. The activations of the widest of them at M = 8,192 are still under half a millisecond at 230 GB/s, so the bound stays the FLOPs at 2,048, 1,818 and 8,192. **Computed.**

Sparse local attention is the 128-wide window: 4·M·16·128·512 FLOPs (QK and AV), no weight traffic. At 2,048 that is 8.589e9 → 0.146 ms; at 1,818, 0.129 ms; at 8,192, 0.582 ms. A compressed layer's top-512 is 4·M·16·512·512: 0.582 / 0.517 / 2.329 ms at those three widths. The indexer scores 2·M·32·n_comp·128 with n_comp = (start + M) / ratio. On a from-start ratio-2 layer that is 0.291 ms at M = 2,048 (n_comp = 1,024) and 0.229 ms at M = 1,818 (n_comp = 909). It grows with the position: the chunk that starts at 6,144 of an 8,192-token prompt has n_comp = 4,096 and costs 1.16 ms. **Computed.** The bytes are the gathered window and the picked latent rows, far under the FLOP time at these widths.

Norms, HC mixes and the sinkhorn (hc_sinkhorn_iters = 20) are elementwise on the chunk. No precise FLOP count is pinned here; they are a few milliseconds a layer, an order of magnitude under the expert traffic. The gate GEMM is plain fp16 [384, 5120]: 2·M·384·5120, 0.136 ms at 2,048 and 0.121 ms at 1,818, compute-bound (the weight is 384·5120·2 = 3.93 MB). **Computed.**

Engram, one layer, this node's wkv: K = (4−1)·(8/4)·256 = 3,072, N = (4+1)·5120/2 = 12,800. FLOPs 2·M·3072·12800: 2.730 ms at 2,048, 2.423 ms at 1,818, 10.92 ms at 8,192. Weight bytes at the assumed k2 = 8 are 3072·12800·0.5 = 19.7 MB → 0.085 ms, so the stage is compute-bound at every width above. Two Engram layers (1 and 14), both before the cut. **Computed.**

Head, one row, pair 0: K = 5120, N = 32,384, k2 = 8. Bytes 5120·32384·0.5 = 82.9 MB → 0.36 ms. FLOPs 2·5120·32384 / 59e12 = 0.006 ms. The quarters of that one row are 1·32384·4·3 = 388 KB of egress. **Computed.** Negligible beside a layer.

Shared-expert bytes in full, since the 32-job re-decode is the only place a weight is read more than once on a 2,048-row chunk. One gate/up matrix pair at k2 = 12 is 2·5120·640·0.75 = 4.92 MB; one down is 1152·2560·0.75 = 2.21 MB. Thirty-two jobs re-read 7.13·32 = 228 MB of weights, 0.99 ms, and the once-only traffic (one weight read, the M-row activations, the scales) is the "weight once" line above. **Computed.**

## Whole prompt

A 2,048-token prompt from the start has replay at 1,920, so its one chunk runs 20 encoder layers at 2,048 rows and then the 128-row decoder tail. A 1,818-token prompt is the same shape at 1,818 rows. An 8,192-token prompt (replay 8,064) is four chunks: the first three end at or before replay and are encoder-only (20 layers, no head), and the fourth runs 20 encoder layers at 2,048 rows plus the 128-row tail. That is 80 encoder layer-passes at 2,048 rows, not one 8,192-row GEMM and not 40 full layers.

Per 2,048-row encoder chunk, **computed**: experts 11.11 ms/layer · 20 = 222 ms (gate/up 6.672, down 3.144, shared 0.878 + 0.419), attention projections 53.5 ms, local window 2.9, compressed scores 10.5, the three indexer layers 0.87 on the first chunk, gate GEMMs 2.7, Engram 5.5. About 298 ms of GPU floor, plus the 142 ms tail on the chunk that crosses replay. Four such encoder chunks are about 1.19 s of that floor, plus one tail, about 1.34 s. The 8,192-row single-GEMM floors (attention projections 10.70 ms/layer, routed gate/up 10.92 ms of FLOPs against 10.16 ms of 2.337e9 bytes, down 4.91 against 5.15) describe a launch the server does not issue.

Collective bytes for one 2,048-row encoder chunk are 4.121 GB (20 · 206.0 MB). The 2,048-token prompt adds the tail's 0.258 GB. The 8,192-token prompt moves 4 · 4.121 + 0.258 = 16.74 GB of all_gather egress. **Computed.** Applying the 3.57 GB/s implied by the 1,818-token measurement (below) gives 1.23 s and 4.69 s of all_gather time. Those two are that rate times these bytes, not a second measurement.

## The 1,818-token prompt

**Measured** (a TP4 run at 2.9 bpw, same activations): RDMA prompt-pass collectives 1,178 ms (all_gather 1,096 + exchange 82) against RCCL 890 ms. RDMA TTFT 1.88 s, RCCL 1.58 s. The 288 ms is the whole TTFT gap. The task brief's ~971 tok/s on the short prompt is the same TTFT (1818/971 = 1.87 s). Remainder after the RDMA collectives: 1.88 − 1.178 = 0.702 s, compute plus host. `coll-compare.txt`'s "prompt pass 40 ms" is a median over short replies (the review, item 3 of the corrections); it is not this prompt.

Cut-aware, from the start: 20 encoder layers at 1,818 rows, then 20 decoder layers at 128 rows.

Encoder GPU floors, **computed**, ×20 except where noted:

| stage | ms/layer | layers | ms |
| --- | ---: | ---: | ---: |
| attention projections | 2.375 | 20 | 47.5 |
| local window | 0.129 | 20 | 2.6 |
| compressed top-512 | 0.517 | 18 (layers 2–19) | 9.3 |
| indexer, ratio 2, n_comp = 909 | 0.229 | 3 (layers 2, 8, 14) | 0.7 |
| gate GEMM 2·M·384·5120 | 0.121 | 20 | 2.4 |
| routed gate/up bytes | 6.541 | 20 | 130.8 |
| routed down bytes | 3.069 | 20 | 61.4 |
| shared, 29 jobs, bytes | 1.170 | 20 | 23.4 |
| Engram GEMM 2·1818·3072·12800 | 2.423 | 2 (layers 1, 14) | 4.8 |

Encoder sum 283 ms. Routed experts are 216 of that. Engram weight bytes at k2 = 8 are 19.7 MB, 0.09 ms, so the GEMM is compute-bound either way the k2 assumption goes.

Decoder tail, **computed as an expectation, not a trace**. 128 rows, top-6, 384 experts: 2.0 routed rows per expert. Under uniform routing the fraction touched is 1 − exp(−2) = 0.865, and n = 128 ≥ 64 so each touched expert is one job. 0.865·(1.258e9 + 5.662e8) / 230e9 = 6.86 ms/layer of expert weights, ×20 = 137 ms. Attention projections ×20 = 3.3 ms. Shared is 2 jobs, about 1.6 ms across the tail. Order 142 ms. The kv compressor runs on the four kv-source layers, and layer 20's update sees the full chunk inside `replayCut`; a few milliseconds, not a separate line. Norms and the HC sinkhorn (20 iterations) are the same: a few milliseconds, order of magnitude. The head is one row. Pair 0's head weight at k2 = 8 is 5120·32384·0.5 = 82.9 MB → 0.36 ms; the FLOPs are 0.006 ms. **Computed.**

GPU floors 283 + 142 ≈ 425 ms. `model.zig` comments the next chunk's first Engram host read (hash, table, decode) at ~0.14 s for 2,048 rows, and `fillSplit` says ~0.1 s at 2,048 rows. That is a comment, not a measured timer on this prompt. Adding 0.10–0.14 s puts the floor near 0.53–0.57 s against the measured 0.702 s remainder. The unexplained piece is about 140–180 ms, roughly 1.3–1.4× the GPU floor once the host read is allowed for. No stage has to be several times off its floor. If the Engram host read is larger than the comment, the GPU is closer to the roof. Routed experts are the bulk of the roof; running them at 160 GB/s instead of 230 adds about 90 ms (216 · (230/160 − 1)). Attention projections at half of 59 TFLOPS add about 50 ms on the 47 ms of encoder projections. Either closes most of the gap. There is no per-stage GPU timer on this path.

## Collectives

Egress is the star: the node sends its part to 3 peers. Equal widths. One 2,048-row layer, **computed**:

- one quarters payload is 2048·2560·4 = 20,971,520 bytes. Attention and MoE, ×3 peers: 125.8 MB.
- wo catRank payload is n·2048·2 (the wider `uw` is 2,048, not the full 4,096) ×3 = 25.2 MB.
- expert catRank payload is n·7·640·2 ×3 = 55.1 MB.

Sum 206.0 MB/layer. That matches the review's item 3 (126 MB of quarters + 80 MB of cat_rank) per 2,048-row layer. Quarters are 125.8/206.0 = 61% of the all_gather bytes; catRank is 39%.

Cut-aware egress at 1,818: 20·eg(1818) + 20·eg(128) = 3.916 GB, with eg(n) = n·(2·2560·4·3 + 2048·2·3 + 7·640·2·3) = n·100608. **Computed.** Divided by the measured 1,096 ms of all_gather this is 3.57 GB/s. That rate is the code's bytes over the measured time, not a measured link rate. A product of 40 full layers would be 7.316 GB and would imply 6.68 GB/s; the code does not move those bytes. At 3.57 GB/s one 1,818-row layer is 51.2 ms (quarters 15.6 ms each sublayer, wo catRank 6.3 ms, expert catRank 13.7 ms). Twenty encoder layers are 1,024 ms and the 128-row tail is 72 ms, which is the measured 1,096 ms. The exchange bucket is Engram: at 1,818 a layer moves n·12800·4 = 93.1 MB to the row partner and n·12800·2 = 46.5 MB to the column partner, ×2 layers = 279 MB. **Computed** bytes; the 82 ms is **measured**.

A 2,048-row encoder chunk is 20 layers, 4.121 GB, 57.7 ms/layer of all_gather at the same implied rate (each quarter 17.6 ms). An 8,192-token from-start prompt is four chunks. If those rows were one GEMM the gate/up floor would be 10.92 ms of FLOPs against 10.16 ms of bytes (2.337e9 bytes) and down 4.91 against 5.15, and attention projections 10.70 ms. The served chunk stays 2,048, so the per-layer rate above is the one that runs, on the encoder layers only.

## What would have to move

Ranked by estimated milliseconds on a from-start prompt. 5 and 6 are separate changes and are not implemented here.

1. **Zip the two halves on world 4** (`fillSplit2d`, this change). The quarters can overlap the other half's prefix or gate/up. At the implied 3.57 GB/s a 2,048-row quarter is 17.6 ms. The overlapping floors are about 3 ms of attention prefix (projections through wo_a, local window, compressed scores; HC mixes are extra and uncounted) and 7.55 ms of gate/up (routed 6.67 + shared 32-job 0.88). The schedule hides two prefixes and two gate/ups per layer: 2·min(17.6, 3) + 2·min(17.6, 7.55) = 21.1 ms/layer. ×20 encoder layers = **422 ms per 4,096-token encoder unit**. **Computed**, and only if `gatherBegin` actually overlaps. If those stages sit on the compute stream, the save is smaller; the hardware check below is what shows which. CatRank cannot overlap: it is the same `Coll` and the same `parts` buffer. ZIP2 of two whole chunks does not re-read expert weights (each chunk is still one 2,048-row pass). A ~7,200-token prompt (measured ~1,074 tok/s RDMA, TTFT ≈ 6.7 s) has replay at 7,072, so ZIP2=1 zips one 4,096-token encoder unit and leaves the rest unzipped: about 0.4 s. ZIP2=2 also zips the following partial (at least 256 rows); that unit is uneven and the extra save is smaller.

   SPLIT alone on the 1,818-token prompt cuts at 896/922 (`splitAt`: replay = 1,690, caps 1,024/1,024, mid rounds to 896). Gather bytes are unchanged. Each half still touches nearly every expert (14 rows/expert, nm = 64, one job), so the routed weights are read twice: extra (1.258e9 + 5.662e8) / 230e9 = 7.93 ms/layer. The half-quarters are ~7.7 ms and the overlap covers about 15 ms/layer, so the net is on the order of 100–150 ms across 20 layers, and it can be ~0 if the expert path is already under 230 GB/s (the second read then costs more than the gather it hides). Turn ZIP2 on for a prompt that spans two chunks. A short prompt's SPLIT may not pay.

2. **Shared expert decoded once per chunk** (pull it onto `exl3.gemm`). At 1,818 the table launches ceil(1818/64) = 29 jobs and re-decodes the k2 = 12 weight each time. The extra weight bytes are 28·(2·5120·640·0.75 + 1152·2560·0.75) = 200 MB → 0.87 ms/layer, ×20 = 17 ms, plus about 1 ms on the tail. **Computed.** Not bit-exact: the K sum order changes and the gate/up epilogue is fused. Not implemented. The old "~30–40 ms" reading is this cost counted on 40 full layers.

3. **A larger prompt chunk** (4,096 / 8,192). `chunk_rows` is a handshake and a cache-key field. Left at 2,048.

4. **Forcing a 256-wide GEMM tile** where the row rule refuses it. The attention GEMMs are already compute-bound at the 59 TFLOPS floor. Skipped.

5. **Pair gate/up 640/512 → 576/576.** Out of scope (the 2D layout). If gate/up is bandwidth-bound the heavy pair saves (640−576)/640 · 6.54 ms ≈ 0.65 ms/layer, ×20 encoder layers ≈ 13 ms. **Computed.** Not a 50 ms item, and not ×40.

6. **catRank as a pairwise exchange.** Out of scope here (the prompt-pipelining change). CatRank is 39% of the cut-aware all_gather bytes, about 430 ms of the 1,096 ms. Sending one peer instead of three would cut that egress to a third, on the order of 0.3 s if the all_gather time scales with those bytes. The peer set is unchanged here.

Kernel occupancy (`PF_GA_GU`, register pressure, nm 64 → 128) cannot be shown on this host, and nm is a numeric change. Not touched. The review's decode items (graphs, copyRows removal, ROUND_MS, the K-split) are not prefill levers.

## What landed

`TF_DS_PREFILL_SPLIT` is legal on world 4. Unset, `split_min` stays 0, `fillSplit` is never entered, `quarters` / `allGather` / `catRank` are the launches the world-4 build runs without it, and no extra GPU memory is allocated for the zip. TP2 (`fillSplit`'s body) and world 1 are unchanged. HIP kernels are unchanged.

When the setting is on and `Engine.two` is set, `fillChunk` takes `fillSplit2d`. The host-tested schedule is `zip2d.zig`: each half's math steps are `fillWhole`'s, a begun gather is closed before any collective, Engram (layers 1 and 14) runs `engramApply` per half after the pending MoE quarters have landed, and the replay cut matches TP2 (A's `replayCut` overlaps B's last MoE quarters, then the second half runs the decoder alone). `quartersBegin` uses the same site and the same byte count as `allParts` (`.two_parts`, layer 0, u8, n·wmax·esize into `parts`). n < 64 does not call `gatherBegin`; `quartersEnd` calls `quarters()`, so the bits match and that gather does not overlap. Prompt halves are larger than that (the split floor on TP2 is 256; ZIP2's partial is at least 256). The post-cut quarters stay `prompt.gather` (a synchronous `allGather`). The head copy is `m.vocab` floats, both pairs' columns.

`tp4.conf` leaves `PREFILL_SPLIT` empty. `PREFILL_ZIP2` still requires the split and the zip, and still defaults off. The handshake `gatherOrder` string does not grow a "2d" token.

## Hardware check

Same prompt, both ranks, two runs. `TF_DS_PREFILL_SPLIT` unset, then `=256`. For a prompt of at least 4,096 tokens also `TF_DS_PREFILL_ZIP2=1` (needs the split and the zip; both ranks alike). `TF_DS_LOGITS_SHA=1` must match the split-off run. That is the bit check.

`COLL_TIME` unset for the TTFT comparison. Sync mode drains around the collective and kills the overlap. With `COLL_TIME=sync`, `prompt_gather` appears (it is 0 today on TP4, because quarters go through `allGather`) and `prompt_gather + all_gather + exchange` should come out at today's `all_gather + exchange` (same bytes). Pre-cut, four `gatherBegin`s per layer (both halves, attention and MoE) is 80 `prompt_gather` calls per 2,048-row encoder chunk. Post-cut quarters stay in `all_gather`. catRank stays in `all_gather`. n < 64 does not `gatherBegin`.

TTFT with the setting on and `COLL_TIME` off, on the ~7,200-token prompt with `ZIP2=1`, should drop by about 0.4 s if the overlap is real. The 1,818-token split is allowed to be small or flat. `TF_GPU_COUNTS=1` with graphs off should show the same kernel ids as `fillWhole` (no new kernels).

The prompts are the 1,818-token one and one of at least 4,096 (the ~7,200-token measured prompt). Do not turn `tp4.conf`'s `PREFILL_SPLIT` on to take the measurement.

What is left of the 0.70 s, after this change, is read with `COLL_TIME=sync` on the same 1,818-token prompt: remainder = TTFT − (all_gather + exchange + prompt_gather). `fillSplit` does not mark phases when a profile timer is on (a profile runs the halves through `fillWhole`), so there is still no per-stage GPU timer on the zip. `TF_GPU_COUNTS` is the launch counter.
