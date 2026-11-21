# TP4 decode attention budget

One node of the 4-bit TP4 lane (16 heads). Measured source: our drafts-on profile, `tf-dsv41-lanes --profile 2`, clock marks, `GRAPHS=0`, 3 requests one at a time, 238 rounds, 3.71 rows a round. Step 48.13 ms, GPU forward 42.93 ms, host enqueue 6.79 ms (the round is GPU-bound; HIP graphs gained nothing), drafter 4.83 ms. Attention phases, ms a round: a_woa 2.89, attn 2.75, a_q 2.42, a_in 1.99, a_x 1.60, mix_attn 1.33, a_core 0.81, a_idx 0.70. Sum 14.49. The two exchange gathers (5.2 ms) are `gather_a` + `gather_m`, outside this sum. Clock marks make the phase total a little long (`PhaseTimer.finish`: each phase is the previous mark to its own).

Hardware, measured: Radeon 8060S, 40 CUs, ~59 TFLOPS fp16, 230 GB/s streaming (`tools/dsv41-hip/exl3/KDEC.md`, bw.hip). EXL3 expert GEMVs on these boxes run at 193–227 GB/s. KDEC's dense numbers below are a different pack (TP2, ~2.9 bpw, often k2=10), cited only as a measured full-GPU rate.

## Phases and kernels

Marks are `round.zig` `attention` then `stretch`. Launch counts below are for 1, 2, 4 and 6 rows: a glinear puts every row in the block (`exl3.hip` `glinear_body`, passes of 16), so the count does not grow with those widths. Grouping is by (k2, wk); a different k2 is another glinear.

| phase | what the mark covers | launches a layer |
|---|---|---|
| mix_attn | `tri_hc.hcPre2` before `attention` | 4 (`_hc_post_only`, `_hc_finish_s`, `_hc_mix_part`, `_hc_sinkhorn`). `TF_HC_CHAIN` is off |
| a_in | `prompt.grouped`: `rot_many` + glinear of wq_a, wkv, and on a kv source the compressor linears when k2 and wk match | 2, or 3 when a compressor k2 differs |
| a_q | `qKvNorm` grid (R, 2 or 3, 1) + glinear wq_b; on an index layer that scores, a second glinear (idx wq_b, wk 4). The wo prefetch is forked on a side stream and not joined here | 2, or 3 when scoring |
| a_idx | compressor (ratio 1: rmsnorm; ratio 2: scatter, gather, compress2, rmsnorm) and the indexer (rope, fp4, `_index_score` or the candidate-pool variant, top-k). Ratio 0 (layers 0–1) runs nothing between the marks | many small, or 0 |
| a_core | `aux.sparse_attn` h16 grid (R, 1, 8) + Triton `_sparse_attn_merge_rot2` grid (R, 16, 1) | 2 |
| a_woa | glinear of this node's 2 wo_a groups, one launch when they share k2 and wk | 1 |
| a_x | `round2d.woExchange` → `catRank`: all-gather + 2 `copyRows`. A slot-sized payload is one `coll.stage` and one `coll.collect` (`gatherStep` sets ncollect = 1). Pair widths are equal (2048, 2048), so no pad copy. R=6 is 6×2048×2 = 24576 B, one slot | 4 |
| attn | glinear wo_b. The next-layer prefetch is queued on the side stream, not waited | 1 |

`attn` is wo_b. `a_core` is sparse attention plus the merge. `a_x` is the cat-rank.

## Shapes on one node

From `config.checkShapes` and `plan.zig` at world 4 (node g is TP2 rank g%2 inside pair g/2, `headSpan` 16). wq_a and wkv are replicated. wo_a is 2 of the rank's 4 groups, each K=4096 (8×512) N=1024. wo_b is K=4096 (4×1024) and N=2560 (half of 5120); `plan_n` stays 5120. wq_b is K=1280 N=8192; `plan_n` stays 16384. The indexer is not head-split: idx wq_b is K=1280 N=4096, replicated, and only on the 8 index sources.

`exl3_linear.plan` (target 192 blocks, min 8 k-tiles a wave) on the columns actually launched:

| linear | sk, wk | blocks | waves (blocks×wk) |
|---|---|---|---|
| wo_a, one group | 8, 4 | 64 | 256 |
| wo_a, the launch (2 groups) | 8, 4 | 128 | 512 |
| wo_b | 8, 4 | 160 | 640 |
| wq_b | 1, 8 | 64 | 512 |
| wq_a | 8, 4 | 80 | — |
| wkv (and each compressor linear) | 8, 4 | 32 | — |
| wq_a+wkv, one launch | 8, 4 | 112 | 448 |
| idx wq_b | 2, 4 | 64 | 256 |

Waves to fill the GPU, from KDEC: glinear holds ~6 waves a SIMD (the lean kernel's 16 is the contrast), 4 SIMDs a CU × 40 CU = 160 SIMDs, 960 waves. wo_a's 512 waves are 512/960 of that. A TP2 wo_a launch is 4 groups, 256 blocks, 1024 waves, and KDEC measured it at 174 GB/s.

## Weight bytes, FLOPs, floor

Trellis bytes are `linearBytes`: K×N×k2/16. The 4.0 bpw pack's per-tensor k2 is not in the tree (recorded glinear plans use both 8 and 10 for these shapes). The table is k2=8 [computed]. Bytes scale by k2/8. suh is fp16 [K], svh fp16 [N]: wo_a's two groups are 2×(4096+1024)×2 = 20480 B, 20480/4194304 = 0.5% of the trellis, left out of the floor.

Weight bytes do not depend on the row count. `strip_tiles_rows` decodes each weight tile once for every row of the pass (`exl3.hip`; R≤8 uses the 8-row part). FLOPs below are one layer at 4 rows, 2×M×K×N, and ×40 / 59e12.

| phase | weight bytes, one layer | ×40 | floor 230 GB/s | measured | FLOPs, R=4, ×40 |
|---|---|---|---|---|---|
| a_woa | 2×4096×1024/2 = 4194304 | 167772160 | 0.73 ms | 2.89 | 0.045 ms |
| attn (wo_b) | 4096×2560/2 = 5242880 | 209715200 | 0.91 ms | 2.75 | 0.057 ms |
| a_q | wq_b 5242880, plus idx wq_b 1280×4096/2 = 2621440 on 8 layers (20971520) | 230686720 | 1.00 ms | 2.42 | 0.057 ms (wq_b) |
| a_in | wq_a 5120×1280/2 = 3276800 + wkv 5120×512/2 = 1310720 = 4587520; plus 4×comp_wkv and 3×comp_wgate, each 1310720 (9175040) | 192675840 | 0.84 ms | 1.99 | under 0.05 ms |
| mix_attn | hc fn fp32 [24, 4×5120] = 1966080 | 78643200 | 0.34 ms | 1.33 | — |
| a_x | the cat-rank payload at R=4 is 4×2048×2 = 16384 B | — | ~0 | 1.60 | 0 |
| a_core, a_idx | no projection weights (KV below) | — | — | 0.81, 0.70 | sparse 0.057 ms |

At k2=10 the a_woa floor is 0.91 ms and the wo_b floor is 1.14 ms [computed, ×10/8]. Both phases stay far above it: effective rate at k2=8 is 167772160/0.00289 = 58 GB/s for a_woa and 209715200/0.00275 = 76 GB/s for wo_b; at k2=10, 73 and 95 GB/s. Experts on the same GPUs are 193–227.

## KV and index bytes

Sparse attention reads the window once (128×512×2 = 131072 B bf16 a row) plus up to `index_topk` 512 FP4 keys. A key is head_dim/2 codes + head_dim/16 scales = 256+32 = 288 B, so 512×288 = 147456 B, 278528 B a row when the top-k is full. `aux_attn.hip` gives each key to one split, so the 8 workgroups do not reread it. Layers 0–1 are window only. ×40 layers at R rows: R×(2×131072 + 38×278528) = R×10846208 B.

| rows | KV bytes, 40 layers | floor 230 GB/s |
|---|---|---|
| 1 | 10846208 | 0.047 ms |
| 2 | 21692416 | 0.094 ms |
| 4 | 43384832 | 0.189 ms |
| 6 | 65077248 | 0.283 ms |
| 3.71 (profile) | 40239432 | 0.175 ms |

The profile's 0.81 ms is ~4× that floor. 288 B gathers do not stream at 230 GB/s. KDEC's TP2 sparse+merge was 0.52+0.13 = 0.65 ms a token at R=1 (16 µs a layer). 0.81 ms / 40 = 20 µs a layer at 3.71 rows, the same kind of number, not a 4× bug. The 16-head kernel (`aux.sparse_attn` h16, `tri_aux.attnFor`) does the same per-head arithmetic as h32 (`aux_attn.hip`). Its LDS comment is 42.7 KB and "three workgroups a WGP"; 42.7×3 = 128.1 KB, so two fit in a 128 KB WGP. At R=1 the grid is 8 workgroups; at 3.71 rows it is ~30, near 2×20 WGPs. That underfill is real at R=1 and is not where the 14.5 ms goes.

Index keys are 128/2 + 128/32 = 68 B a position, shared by the 32 index heads. `nb` is the bucket / the layer's ratio. `idx_all` (nb ≤ 512 and not the candidate source) is an iota and reads no keys. Past the candidate source the pool path scores 2048×8 = 16384 keys, not the whole bucket. Upper bound, one row, the layers that score:

| context | who scores | bytes a row | floor at R=4 |
|---|---|---|---|
| 1k | ratio-2 sources 2, 8, 14 have nb≈500 and take idx_all. Five ratio-1 sources score ~1000 | 5×1000×68 = 340000 | 0.006 ms |
| 8k | 3 ratio-2 layers × 4000 and 5 ratio-1 × 8000 (under the 16384 pool) | 3536000 | 0.061 ms |
| 32k | 3×16000, the candidate source ×32000, four later layers ×16384 | 9896448 | 0.172 ms |

Index FLOPs at 32k, R=4, that same key count: 2×4×32×128×(3×16000+32000+4×16384) / 59e12 = 0.081 ms. a_idx is 0.70 ms averaged over all 40 layers, and layers 0–1 do none of it. It is launch latency on a few layers, not the attention budget.

## What sits far above the floor

a_woa, attn and a_q. The cause is `plan`'s min-8-tiles rule on a halved N.

TP2's wo_a reads four groups (KDEC: 10.5 MB, 2.41 ms, 60 µs a layer, 1024 waves, 174 GB/s). This node launches two groups, 4.19 MB at k2=8, and the profile spends 2.89/40 = 72 µs. wo_b is the same story: KDEC 2.72 ms for the full N=5120, the profile 2.75 ms for N=2560. The K chain is 256/(8×4) = 8 tiles a wave, and the launch has 512 or 640 waves against 960. `plan` will not take the next split: sk 16 leaves 4 tiles a wave, under its minimum of 8, and wq_b's sk stays 1 because N is planned as the TP2 16384 (80 tiles, wk 8, already 10 a wave). So each glinear keeps TP2's wall time on about 0.4× the bytes.

A full-GPU fixed-cost model, computed from KDEC's measured rates (bytes/174 GB/s + ~15 µs for wo_a, bytes/193 + ~12 µs for wo_b), is (24+15) µs × 40 = 1.56 ms for a_woa and (27+12) µs × 40 = 1.57 ms for wo_b. The profile is 2.89 and 2.75. The extra is the underfilled chain, not a second read of the weights and not the prefetch: the wo prefetch is forked at the end of a_q and joined only at the end of the stretch.

a_x at 1.60 ms is 40 µs × 40 layers of cat-rank latency on a 16 KB payload. That is the largest gap that does not change a sum. It is smaller than the projection gap, and the round trip is most of the 40 µs.

Not the cause, from the code: a per-row re-read (the strip kernel decodes a tile once); a per-head sparse launch (one workgroup does all 16 heads); fp32 weights (the trellis is the k2 pack; the fp32 partials are the split-K Z, tiny); the h16 sparse path (same head arithmetic as h32, and a_core matches TP2's measured sparse time). A per-row launch would be bit-identical and would reread the weights R times. At 72 µs and R=4 it needs 4194304×4 / 72e-6 = 233 GB/s to break even, above the streaming ceiling. KDEC's extra K splits were slower only on launches that already filled the GPU. These do not.

## The change

`TF_DS_TP4_FILL=1` (default off) sets `Linear.sk_fill` on a 2D node's wq_b, wo_a and wo_b only, and only when the projection is not dense. `planFor` then raises K splits for a window of 1..16 rows: sk doubles while the column blocks are under 192, the next split divides K, and a wave keeps at least 4 tiles. wk is unchanged, so the kernel specialisation and the (k2, wk) grouping stay. No HIP source changes: `L.SK` was already a runtime field.

| linear | filled sk | blocks in the launch | waves | tiles a wave |
|---|---|---|---|---|
| wo_a, 2 groups | 16 | 256 | 1024 | 4 |
| wo_b | 16 | 320 | 1280 | 4 |
| wq_b | 2 | 128 | 1024 | 5 |

sk 32 on wo_a would be 2 tiles a wave and is refused. wq_b cannot take sk 4 (80 % 32 ≠ 0). A full TP2 shape (wo_b N=5120, 320 blocks) does not grow even if asked. m>16, including `TF_DS_ROUND_ROWS=48` and the wide prefill path, returns `plan()`. The drafter loads with `pair = null` (`plan.wants`), so it never sets `sk_fill`. TP2 and world 1 have no pair, so the env does not touch them either.

The new order is glinear's: each wave sums its k tiles left to right, waves add in wave order, then splits add in split order (`foldTiles`). Half-wave adds stay inside a tile. That is a different fp32 association, so the setting is off unless asked, and logits will not match today's SHA while it is on. With it unset, `planFor` is `plan`, `zFloats` is the old count, and the launch sequence is today's. `zFloats` at the chunk's 128 rows does not grow: wo_b's base partials are 8×128×2560 = 2621440 floats, and a filled wq_b needs 2×16×8192 = 262144.

## Hardware check

Same profile as above (`TF_DS_CLOCK_MARKS`, `--profile 2`, `GRAPHS=0`), with `TF_DS_TP4_FILL=1` on every rank. a_woa should move from 2.89 ms toward ~1.6 ms (256 blocks, 4 tiles a wave, the same bytes) and attn from 2.75 toward ~1.6. a_q should drop with wq_b; it also holds `qKvNorm` and, on scoring layers, the indexer glinear, which this setting does not retile. Launch count does not change (grid.x does: wo_a 128→256, wo_b 160→320, wq_b 64→128), so the confirmation is the phase ms, not a launch counter. If a_woa stays near 2.89, the extra splits did not hide the K chain and the setting stays off.

## Exact order

`TF_DS_TP4_FILL=exact` is the same three linears (a 2D node's wq_b, wo_a, wo_b, not dense), default off. `sk_fill` is 2. `planFor` returns `plan()`: the sk and wk a TP2 rank uses for that column (`plan(k, plan_n)`). The fp32 order does not move. The kernel changes, and only at 1..16 rows. `TF_DS_TP4_FILL=exact_lds` is `sk_fill` 3: the same plan and the same fp32 order, on `exl3.glinear_mlp_lds`.

`exl3.glinear_mlp` and `exl3.glinear_mlp_lds` (`zig/kernels/hip/dsv41/exl3_mlp.hip`) are the two families. `exl3.hip` only gains one grouped-expert instantiation (`TF_DS_EXL3_GROUPED(8, 12)`, for the 4.0 bpw pack); its existing kernels are unchanged. The dot2 chain, the half-wave add, the wave add and the split add are the lines in `exl3.hip` at 310-325, 371-407, 616 and 655-695, copied verbatim and not unrolled across depth. `lds_fence` still waits `lgkmcnt`, not `vmcnt`. m>16, including the wide prefill, stays on `exl3.glinear`. An older kit without a family's symbols still loads. A rank with `exact` set and `glinear_mlp` missing refuses to open. `exact_lds` uses `glinear_mlp_lds`, and a `(k2, wk)` whose LDS symbol is absent uses the register symbol. `TF_DS_MLP_SCAN` (not defined for the kit, not in `registry.json`) keeps the other depths for the next pass.

The other levers do not fit this grid. glinear is ~231 VGPRs, 6 waves a SIMD, 960 waves on 40 CUs. These launches are 512, 640 and 512 waves, 3.2 to 4.0 a SIMD, already resident, so fewer VGPRs add no waves. The two wo_a groups are already one glinear when they share k2 and wk. wo_b is the only linear in `attn`. A second stream adds a launch and no waves. More K splits cannot be exact: a wave's k tiles are one fp32 chain. That is `=1`, which changes logits.

One k tile at k2=8 is 32 lanes × 8 dwords × 4 = 1024 B. wo_a's 512 waves already have 524288 B in flight at depth 1. The bandwidth-delay product at 230 GB/s and 400 ns is 230e9 × 400e-9 = 92000 B, so depth 1 covers it when the loads stay in flight. If the compiler drains `vmcnt(0)` before each use, depth 4 collapses to today's depth 1. KDEC's pd2 (two k steps) was noise on a full GPU, and a small grid holds a lower sclk (the wide-kernel note in `exl3.hip`). This stays off until a measured number moves.

| linear | sk, wk | blocks | waves | tiles a wave |
|---|---|---|---|---|
| wo_a, 2 groups | 8, 4 | 128 | 512 | 8 |
| wo_b | 8, 4 | 160 | 640 | 8 |
| wq_b | 1, 8 | 64 | 512 | 10 |

## Measured, exact depth 4

4-bit TP4, replies 8/8 byte-equal to base. Slower: drafts off 36.3 vs 36.9 tok/s (−1.7%), 8 chats 168 vs 176 (−4.6%). `=1` was +5.3% drafts off and +10.4% with code drafts. hipcc `-Rpass-analysis=kernel-resource-usage`: the depth-4 ring (dot body unrolled once per slot) is 253–256 VGPRs, scratch on 10 of 12 variants, occupancy 5. Original `glinear` is 217–225 VGPRs, 0 scratch, occupancy 6.

## Scan and the pick

`exl3_mlp.hip` takes `DEPTH`, `ACT` and `STAGING` as template arguments. The k loop is not unrolled, so the dot2 chain stays one copy. `-DTF_DS_MLP_SCAN` (not the kit build, not in `registry.json`) still emits `tf_ds_exl3_glinear_mlp_scan_k{4,6,8,10,12,16}_w{4,8}_{d2,d3,d4,d4a1,d2a1e6,lds2,lds2e6}` and `_lds4` except `k10_w8`, `k12_w8`, `k16_w8`. The kit build does not define the macro. The TheRock 10.2 table picked, for each `(k2, wk)`, the deepest schedule with 0 spill and 0 scratch. Register picks also require occupancy ≥ 6. `k16_w8` has no such register schedule: depth 2 with both tiles in registers is 250 VGPRs, occupancy 5, and `amdgpu_waves_per_eu(6)` spills 12 B/lane, so the unpinned depth-2 schedule ships. Every LDS variant reaches occupancy ≥ 6, so none falls back. `allow_scratch` is off on both families. Logits of either setting must match today's SHA. Do not compare them to a `=1` run.

`d2`/`d3`/`d4` keep that many weight and activation tiles in VGPRs. `d4a1` is four weight tiles and one activation. `d2a1` is two weight tiles and one activation. `lds2`/`lds4` park that many weight tiles in LDS (gfx1151 has no `global_load_lds`). `e6` is `amdgpu_waves_per_eu(6)`.

| symbol | config | VGPRs | occ |
|---|---|---|---|
| `glinear_mlp_k4_w4` | d4a1 | 234 | 6 |
| `glinear_mlp_k4_w8` | d4a1 | 236 | 6 |
| `glinear_mlp_k6_w4` | d2 | 235 | 6 |
| `glinear_mlp_k6_w8` | d2 | 237 | 6 |
| `glinear_mlp_k8_w4` | d2 | 236 | 6 |
| `glinear_mlp_k8_w8` | d2 | 238 | 6 |
| `glinear_mlp_k10_w4` | d2a1 | 234 | 6 |
| `glinear_mlp_k10_w8` | d2a1 | 236 | 6 |
| `glinear_mlp_k12_w4` | d2a1 | 238 | 6 |
| `glinear_mlp_k12_w8` | d2a1 | 239 | 6 |
| `glinear_mlp_k16_w4` | d2a1e6 | 240 | 6 |
| `glinear_mlp_k16_w8` | d2 | 250 | 5 |
| `glinear_mlp_lds_k4_w4` | lds4 | 210 | 7 |
| `glinear_mlp_lds_k4_w8` | lds4 | 212 | 7 |
| `glinear_mlp_lds_k6_w4` | lds4 | 213 | 7 |
| `glinear_mlp_lds_k6_w8` | lds4 | 215 | 7 |
| `glinear_mlp_lds_k8_w4` | lds4 | 210 | 7 |
| `glinear_mlp_lds_k8_w8` | lds4 | 212 | 7 |
| `glinear_mlp_lds_k10_w4` | lds4 | 213 | 6 |
| `glinear_mlp_lds_k10_w8` | lds2 | 214 | 7 |
| `glinear_mlp_lds_k12_w4` | lds2 | 212 | 7 |
| `glinear_mlp_lds_k12_w8` | lds2 | 214 | 7 |
| `glinear_mlp_lds_k16_w4` | lds2 | 210 | 7 |
| `glinear_mlp_lds_k16_w8` | lds2 | 212 | 7 |

Scratch is 0 on every row. Symbol names in the kit are `tf_ds_exl3_` plus the name above.
