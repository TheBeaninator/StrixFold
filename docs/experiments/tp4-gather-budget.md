# TP4 decode gather budget (gather_a, gather_m)

World 4, node g = TP2 rank g % 2 inside pair g / 2. `gather_a` and `gather_m` are `prompt2d.Two.quarters` of the fp32 partials (`round.zig` stretch: `quarters(pa, ga, R, ow, 4)` and `quarters(pm, gm, R, dw, 4)`). `ow == dw` (`Two.init` returns `UnequalPartialSplits` otherwise). Served widths (`plan.zig` `cut` of `hidden` 5120, and `down0` 20 blocks): both pairs 2560 columns. Engram quarters (`ew`, layers 1 and 14) and the head (`hw`) are other phases. `catRank` (wo exchange `a_x`, expert intermediate `m_x`) is a pair exchange with g ^ 2, not these two phases.

Every number is **computed** (arithmetic here) or **measured** (the served build, eager, `--profile 2`, `TF_DS_CLOCK_MARKS=1`, rank 0). Link rate ~4.5 GB/s is our measured 100 GbE rail, not a rate from this note. On-node DRAM floor 230 GB/s is the prefill note's streaming figure.

## Which clocks and launches the profile counts

`Round.mark` adds `gpu.launch.count` since the previous mark, then launches one `gpuClock`. A phase's launch count therefore includes the previous phase's clock kernel plus the kernels before this mark. `round_launches` is that host count over every round (uncapped). `round_clock_us` is the GPU gap ending at that clock.

Clocks attempted in one forward, **computed** from `stretch` / `attention` / `moe` on the 2D path (shared expert in the table, so `m_rt`, `m_gu`, `m_x` all mark). A non-engram layer is 15 phase marks: mix_attn, a_in, a_q, a_idx, a_core, a_woa, a_x, attn, gather_a, mix_ffn, m_rt, m_gu, m_x, moe, gather_m. Engram layers add the engram mark (16). Stretches cut before layers 1 and 14: layer 0; layers 1..13; layers 14..39. Phase marks: stretch 0 = start + embed + 15 + post = 18; stretch 1 = 16 + 12·15 + post = 197; stretch 2 = 16 + 25·15 + post + head = 393. Sum 608. Plus `clock_start` and each stretch's `clock_wait` and `clock_tail`: 7. Total 615. `max_clocks` is 512, so 103 clocks drop off the tail: `clock_tail`, head, post, all of layers 34..39 (6·15 = 90), and the last 10 marks of layer 33 (from a_woa through gather_m). `gather_a` and `gather_m` are kept for layers 0..32 only: 33 marks a round. **Computed.** That is the measured "33 marks a round each". Full-round GPU time of these phases is the clock figure · 40/33. Launch counts are all 40 layers.

`PhaseTimer` (`--profile 2`) records phase events only, cap `max_marks` 1024. 608 phase marks fit, so `phases_ms` is all 40 layers. **Computed.** It is a different timer from the clocks, not the same 512 cap.

## The collective, per layer, per row count

`quarters` (`prompt2d.zig`): rings are compiled out on HIP (`verbs_rdma` false). Then:

- R = 1, `w[0] == w[1]`, `comm.swap12`, payload ≤ `part_bytes`: one `allGather` with `Site.swap12`. `Word.place` = 2·(g%2) + g/2, so blocks land g0, g2, g1, g3 straight into `dst` [2, 1, 2W]. No `copyRows`. **This is the measured one-row path** (launch count below). `COLL_SWAP12` is on in that profile.
- Otherwise: `allParts` (pad `copyRows` only when the pairs differ, then one rank-order `allGather` into `parts` [4, R, wmax]) plus `scatterQuarters` (4 `copyRows`, offsets `zip2d.quarterAt`).

`ow` and `dw` are equal, so gather_a and gather_m take no pad copy. Head pairs are 253 and 252 blocks of 128 (32384 and 32256 columns): unequal, so the head stays on the four copies even at R = 1. Engram pairs are each 12800 columns (**computed**: `(hc+1)·5120 = 25600`, 200 blocks, mid `(200+1)/2`): equal, so Engram follows the same branch as gather_a. It is not inside these two phases.

One slot. `gatherChunk` = alignDown((slot − 4·64) / 4, 64). At the serve `COLL_SLOT_KIB` 4352, slot = 4456448, chunk = 1114048 bytes (**computed**). A rank's gather_a block is R·2560·4 = R·10240 bytes. 1114048/10240 = 108.8, so R ≤ 108 is one chunk. R = 1, 2, 4, 6, 8 and the measured 5.49 are one chunk. A one-chunk call uses lane 0 only (`laneOf(0)`; `TF_COLL_LANES` does not pipeline it). `gatherStep` with one chunk: one `coll.stage`, one `coll.collect` of `world·bytes` from the slot start (`ncollect = 1`). No reduce-scatter.

Why R > 1 cannot reuse swap12's straight write. `dst` is [2, R, 2W]: row `i` of rank `r` is pair 0's W then pair 1's W. swap12's collect writes four contiguous [R, W] blocks. For R = 1 those four blocks are the destination. For R > 1, pair 1 of a rank sits at a stride of 2W inside the row, not at the end of that rank's pair-0 block. A contiguous stage/collect (`coll.hip` copies a byte range) cannot place it. Four per-row collects would be 4·R launches.

## Bytes, wire, launches

Per rank, per gather, per row, equal 2560 fp32 columns. **Computed.**

| | send | recv payload | trailers each way | egress | wire at 4.5 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| one row | 10240 B | 3·10240 = 30720 B | 3·64 = 192 B | 3·(10240+64) = 30912 B | 6.869 µs |
| R rows, one slot | R·10240 | 3·R·10240 | 192 B (once) | 3·(R·10240+64) | below |

Wire µs, one gather, **computed** as egress/4.5e9: R=1 6.869, R=2 13.683, R=4 27.349, R=6 41.003, R=8 54.677. R=5.49: 3·(5.49·10240+64) = 168845 B → 37.52 µs. Both gathers, 40 layers: R=1 is 80·6.869 µs = 0.550 ms; R=5.49 is 80·37.52 µs = 3.002 ms. The NIC is shared by the three QPs; full duplex, so the time is the egress (ingress equals it). Own block is staged and collected locally and is not on the wire.

Launches inside the phase window, one layer, **computed**: the previous mark's clock, plus stage, plus collect, plus copies.

| R | path | device kernels | launches in the phase | ×40 layers |
| --- | --- | --- | ---: | ---: |
| 1, swap12 | straight into dst | stage, collect | 3 | 120 |
| ≥2, equal, one slot | allParts + 4 copyRows | stage, collect, 4 copies | 7 | 280 |

`copyRows` of a 16-byte-aligned row is `tf_ds_glue_copy_rows16` (one launch, grid.y = R). Four of them. No other assembly kernel.

**Measured** launches: drafts off, 1.04 rows, gather_a 121 and gather_m 121. **Computed** 120. The +1 a round is 0.025 a layer, not a second row: a 4% share of 2-row rounds would add 40·4·0.04 = 6.4 launches. Drafts on, 5.49 rows: 280 and 280, exactly 40·7. Every drafts-on gather took the multi-row path and one slot. The four extra launches a layer are `scatterQuarters`.

`copyRows` bytes at R=5.49: 4·5.49·10240 = 224870 B, 0.98 µs at 230 GB/s (**computed**). The cost is the launches, not the DRAM.

## Reconciling the measured times

Clocks are 33 layers. Per recorded layer, **computed** from the measured ms:

| | gather_a | gather_m | wire | above wire |
| --- | ---: | ---: | ---: | ---: |
| 1.04 rows, clock 1.04 / 1.09 ms | 31.5 µs | 33.0 µs | 6.87 µs | ~25 µs |
| 5.49 rows, clock 5.06 / 3.04 ms | 153.3 µs | 92.1 µs | 37.5 µs | 116 / 55 µs |

Phase events (all 40 layers, **measured** 1.27+1.33 ms and 5.78+3.74 ms): per layer 31.8 / 33.3 µs at 1.04 rows, and 144.5 / 93.5 µs at 5.49. The one-row clocks and phase events agree once the 33/40 cap is applied (1.04·40/33 = 1.26 ms vs 1.27; 1.09·40/33 = 1.32 vs 1.33). gather_m at 5.49 agrees too (3.04·40/33 = 3.68 vs 3.74). gather_a's clock sits a little above its phase average.

The ~25 µs at one row is the fixed part of one slot gather (stage, the proxy's RDMA and three trailers, collect, the clock). **Computed** as measured gap minus wire. It is the same on both phases.

At 5.49 rows the gather_m growth above that fixed part is 92.1 − 31.5 − (37.5 − 6.87) ≈ 30 µs (**computed**). Four `copyRows` at ~7 µs, plus the larger stage/collect, accounts for it. gather_a's extra ~60 µs is not the copies (they are the same kernels on both phases). The all-gather waits for the slowest rank; gather_a follows attention, gather_m follows the MoE. That skew is inside the collective and is not a copy.

Full round, clocks scaled by 40/33 (**computed**): 1.26+1.32 = 2.58 ms at one row against a 0.55 ms wire floor; 6.13+3.68 = 9.82 ms at 5.49 rows against a 3.00 ms wire floor. Forward is **measured** 29.3 ms and 55.4 ms. The two phases are 5.06+3.04 = 8.10 ms of the 55.4 ms clocked round, 15% of the clocks that were kept; the task's 17% is the phase events (5.78+3.74)/55.4.

Prefill, 26 tokens, one chunk (**measured** gather_a 10.4 ms, gather_m 10.3 ms) is not this floor. n=26 still calls `quarters` (the zip begin path starts at n≥64) but the payload is 26·10240 = 266240 B, wire 177 µs a gather (**computed**), and the pass is many layers of that plus the same fixed cost. Decode R≤16 is the table above.

## Ranked exact changes

Estimated milliseconds are a round at R≈5.5, both phases, 40 layers. Not measured.

1. **Assemble the multi-row equal-width gather with two `copy_rows16` instead of four. Same rank-order `allGather`, same site, same bytes.** Per pair p (not per TP2 rank): ranks g = 2p and g = 2p+1 are adjacent in parts [4, R, W]. This gather is not swap12, so `Word.place` is the rank; swap12's permutation is the one-row path only. `copyRows(src = parts + 2p·R·rb, src_ld = rb, dst = dst + p·rb, dst_ld = 2·rb, bytes = rb, rows = 2R)`. Source row r·R + row is rank g = 2p+r and lands on destination row r·R + row at column block p, which is `quarterAt`. `copy_rows16` is 16-byte words, grid (ceil(n16/256), rows), one word a thread, no per-thread loop. At R=5.49, rb=10240, n16=640, grid (3, 11) a launch (**computed**), 2·5.49·10240 = 112435 B a launch; two launches move the same 224870 B the four copies did (0.98 µs at 230 GB/s). The byte strided kernel is the wrong one: one block an outer group, a byte a thread, a u64 div/mod per byte. outer=2 is 2 blocks and ~220 dependent iterations a thread at this size, tens of µs a launch, which would wipe the saving. Save is still 2 launches · ~7 µs · 80 gathers ≈ **1.2 ms** (the ~7 µs is the gather_m remainder ÷ 4), and the two that remain are copy16. On the 33-layer clock that is ~1.0 ms. R=1 stays on swap12. Head stays on four `copyRows` (unequal widths). The only gate is copy16's own: addresses and rb a multiple of 16, and rows = 2R ≤ 65535 (grid.y). No R≤16 cap. Wire floor unchanged.

2. One fused scatter kernel instead of four `copyRows`: save 3 launches, ~1.7 ms, and it needs a new HIP kernel. Larger than (1), not taken while (1) is host-only and uses the kernel the kit already loads.

3. Overlap gather_a / gather_m with compute that does not depend on them, as `TF_DS_PREFILL_ZIP2` does for a prompt half. Decode has one window. The consumer of `ga` is the next `hcPre2` (`mix_ffn`); the consumer of `gm` is the next layer's `mix_attn`. Nothing on the main stream between the producer GEMM and that mix is independent of the gather. Sinkhorn of `mix_attn` already runs during attention and is joined before gather_a. **~0 ms.** Prefill zip is already shipped.

4. Fuse or batch the per-layer gathers. Each layer's partial exists only after that layer's wo_b or down GEMM, and the next mix consumes it before the next layer. **0 ms.**

5. Pipelined lanes on these messages. A one-slot gather is lane 0. Splitting a gather whose fixed part is ~25 µs and whose wire is ~7 µs a row into several one-row round trips adds fixed parts. **~0 or slower.** Lanes stay the path for a gather that already exceeds a slot.

6. Narrower fp32. wo_b and down outputs are arbitrary fp32 sums. A bf16 or fp16 store is not bit-identical. **Not exact. Do not.**

(1) is implemented behind `TF_DS_VGATHER` (below). It does not change a sum, an order, or a byte the rank stores. Default off leaves the 4-`copyRows` path, including its launches.

## Implemented: TF_DS_VGATHER

`VGATHER=0` in serve.conf (default). `1` exports `TF_DS_VGATHER=1` to every rank (`serve_lib.sh`, the same shape as `TP4_FILL`). The world-4 collective reads it at open (`coll.vgatherFromEnv`: the string `1` only; anything else, including unset, is off). World 2 and world 1 leave the flag false. It is not in the RDMA handshake: the all-gather word, site and bytes are the rank-order gather issued with it off, so a rank with the key on still matches a rank with it off on the wire. No HIP source changed. The two launches are `ops.copyRows`, which takes `tf_ds_glue_copy_rows16` when the addresses fit it and the byte strided kernel otherwise. This path calls it only when they fit, and falls back to the four copies when they do not.

With the key on, after the rank-order parts have landed (`quarters` and a begun prompt gather's `quartersEnd`), equal widths and n > 1: `placeQuarters`, two `copyRows` as in (1). R = 1 with `COLL_SWAP12` stays the straight gather (zero copies) with the key on or off. R = 1 without swap12, unequal widths (the head), a row byte count that is not a multiple of 16, and n > 32767 (so 2n would exceed copy16's grid.y of 65535) stay on four `copyRows`. Engram quarters are equal-width and 16-byte aligned, so they take the two copies too; that phase is not gather_a or gather_m. A prompt chunk does too when the flag is on. Arena `take` is 256-byte aligned, and rb = 10240, so the served gather_a / gather_m hit copy16.

Launches in the phase window, multi-row equal, one slot, key on, **computed**: previous clock + stage + collect + 2 copy16 = 5 a layer, ×40 = 200. The four-copy path is 280 (**measured** at 5.49 rows). The drop is 80 a phase. One row stays 120 (**computed**) / 121 (**measured**). Each copy16 at R=5.49 is the grid in (1): 640 words, 3 by ~11 blocks, one 16-byte word a thread.

Host, not hardware: `prompt2d_test` "world 4: TF_DS_VGATHER quarters match the four copies, R = 1..8" — on and off write the same bytes for R = 1..8, default off is four copy16, on is two copy16 and zero strided for R = 2..8 (and for R = 17). R = 1 with swap12 is one gather and zero copies either way. Unequal widths stay four copy16. A 4-byte row (not a multiple of 16) falls back to four strided copies, not two. The world-4 fault matrix (`rdma_proxy4_test`) was run with `TF_DS_VGATHER=1`. Every case passed except the laptop-flaky "rank 0 names the wrong peer" (check.sh's baseline failure on this machine; this run: Mismatch on ranks 0 and 1, PeerLost on 2 and 3). The matrix drives the proxy steps, not `quarters`, so the key cannot change a step.

## Hardware check

Eager, `--profile 2`, `TF_DS_CLOCK_MARKS=1`, rank 0, the same draft settings as the measurements above. `VGATHER=1` against `VGATHER=0`.

- `round_launches` gather_a and gather_m at ~5.5 rows: 280 → 200 (or 201 if the one-row profile's extra launch is still present). Still 280 means the key did not take this path. One row (drafts off, swap12): still ~121. Not down to the one-row count: the clock, the stage and the collect remain. The two launches that stay are copy16, not the byte kernel: at R≈5.5 each is grid (3, ~11), 640 16-byte words a row, one word a thread, ~112 KB (**computed** in (1)).
- `phases_ms` gather_a + gather_m (all 40 layers): about 1.2 ms less a round at R ≈ 5.5 (**computed**: 2 launches × ~7 µs × 80 gathers; the ~7 µs is the gather_m remainder ÷ 4, not a separate measurement). `round_clock_us` of the two phases (33 of 40 layers): about 1.0 ms less combined (1.2 × 33/40). One row: both timers unchanged past noise. A byte-strided copy would show the time going up, not down. gather_a staying above gather_m is the rank skew inside the all-gather; these copies do not remove it.
- Bytes are the host test above. The hardware profile does not check them.
