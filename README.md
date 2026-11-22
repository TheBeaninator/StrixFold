# StrixFold

[![Discord](https://img.shields.io/badge/Discord-StrixFold-5865F2?logo=discord&logoColor=white)](https://discord.gg/D4FgAJnKd9)
[![X](https://img.shields.io/badge/X-@Shoestring__Dev-000000?logo=x&logoColor=white)](https://x.com/Shoestring_Dev)

TensorFold for AMD Strix Halo: DeepSeek-V4.1-Flash on two Strix Halo boxes, split across both over plain USB4
networking.

This is a fork of [TensorFold](https://github.com/ashhart/TensorFold) by way of
[Bertholomus AI's DeepSeek fork](https://github.com/bertholomus/TensorFold). It adds an AMD HIP backend, kernels
written for Strix Halo (gfx1151), and tensor parallel across two boxes over their USB4 ports. On two Ryzen AI Max+ 395
machines it runs DeepSeek-V4.1-Flash at about 72% of the speed of two NVIDIA DGX Sparks, with the same output quality.

![One chat: 52.8 tokens per second, against 13.3 for the previous engine and about 73 for two DGX Sparks](docs/img/one-chat.svg)

![Prompt reading: 1,543 tokens per second, against about 270 for the previous engine and 2,180 for two DGX Sparks](docs/img/prompt-reading.svg)

![Several chats at once: from 52.8 tokens per second for one chat to 108 for eight](docs/img/several-chats.svg)

## Results

| | One chat | Prompt reading | Eight chats at once |
|---|---:|---:|---:|
| DwarfStar, the previous two-box engine (18 Oct 2026) | 13.3 tok/s | ~270 tok/s | |
| This port, first run (19 Oct) | 20.6 tok/s | 829 tok/s | |
| This port now | **52.8 tok/s** | **1,543 tok/s** | **108 tok/s** |
| 2 × DGX Spark, same weights (reference) | ≈73 tok/s | 2,180 tok/s | |

- One chat is 72.3% of two DGX Sparks; prompt reading is 70.8%.
- Writing speeds are with drafting on, the shipped default (`DRAFTS=1` in `serve.conf`). With it off, one chat writes
  about 32 tok/s. Drafting gives exactly the same text either way.
- Writing speed holds at long context: from an empty prompt to 48K tokens of context it drops about 4%
  ([Speed at long context](docs/RESULTS.md#speed-at-long-context)).
- Quality matches the Spark build: MMLU-Pro 84.8%, HumanEval+ 92%.
- Context windows from 16K up to 1M tokens.
- How each number was measured, with the raw tables: [docs/RESULTS.md](docs/RESULTS.md).
- Numbers from other people's boxes: [docs/COMMUNITY-RESULTS.md](docs/COMMUNITY-RESULTS.md). Add yours.

New boxes? [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md) goes step by step from two AMD Ryzen AI Halo boxes
or two Bosgame M5s to a first answer and a benchmark.

## Tested on

| | |
|---|---|
| Boxes | 2 × AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB unified memory each |
| Link | two USB4 cables (thunderbolt_net), MTU 1500 on both |
| GPU stack | ROCm 10.0.0-4 (HIP 7.15, clang 23), Linux kernel 7.2.7 |
| Toolchain | Zig 0.17.0, Python 3.13 |
| Model | DeepSeek-V4.1-Flash-1017 as an EXL3 2.9 bpw pack (197 GB), plus its Engram shards (203 GB). Not included: download them yourself. |

No NVIDIA driver, CUDA or NCCL is used. Text is the default; image input runs on the GPU with `MODALITIES=text_image`
(the model's vision tower on the first box; setup and limits in
[tools/dsv41-hip/ADAPTER.md](tools/dsv41-hip/ADAPTER.md) 8.1). Images can also go through a small local vision model:
[docs/LOOK-AT-IMAGE.md](docs/LOOK-AT-IMAGE.md).

## What this fork adds

- An AMD HIP backend for the DeepSeek-V4.1 family. The branch this fork started from had no HIP code; TensorFold's main
  line has its own HIP backend, and the DeepSeek and two-box USB4 changes are being moved onto it as small pull requests
  ([#2065](https://github.com/ashhart/TensorFold/issues/2065), [#2066](https://github.com/ashhart/TensorFold/issues/2066)).
- Strix Halo kernels for the EXL3 weights, sparse attention, top-k and the routed experts.
- Tensor parallel across two boxes over plain USB4 networking, with prompt reading running while data moves between
  them, and the traffic split over both cables.
- Drafting that gives exactly the same text with drafts on or off, and switches itself off when six or more chats are
  running.
- Long context up to 1M tokens, and a fix for a crash between the two boxes on long prompts.
- Output checks (logits hashes and decode receipts) and the benchmark tools, so you can verify your own build.

The full list, file by file, is in [CHANGES.md](CHANGES.md).

## Build and run

The commands this port runs, in order (build, prepare the inputs, serve, check, benchmark), are in
[tools/dsv41-hip/BUILD_AND_RUN.md](tools/dsv41-hip/BUILD_AND_RUN.md). From new boxes, start with
[docs/GETTING-STARTED.md](docs/GETTING-STARTED.md): it puts the setup guides and these commands in order.

On a Bosgame M5 (same chip, ships with Windows), first follow [docs/BOSGAME-M5.md](docs/BOSGAME-M5.md): BIOS, Debian
13, the 7.2.7 kernel with Secure Boot on, ROCm 10.0, and the fans for long loads.

Driving it from an agent harness (tool calls, and keeping agents out of repeat loops):
[docs/AGENTS.md](docs/AGENTS.md).

Images: natively with `MODALITIES=text_image` (tools/dsv41-hip/ADAPTER.md 8.1), or for a text-only server through a
small local vision model that describes them and answers follow-up questions:
[docs/LOOK-AT-IMAGE.md](docs/LOOK-AT-IMAGE.md).

## Tuning the two boxes

What made the link between the boxes fast and steady (MTU, which traffic goes on which cable, TCP retransmit floors,
power settings) is in [docs/TUNING.md](docs/TUNING.md). Most of it applies to any two-box Strix Halo setup.

## Four boxes (TP4), on this branch

This branch adds a four-box split: DeepSeek-V4.1-Flash at 4.0 bpw, tensor-parallel over four Strix Halo boxes (the
exact 2D split `prompt2d.zig` describes, world 4). It is the fork's working build, offered upstream in
[issue #57](https://github.com/Shoestring-dev/StrixFold/issues/57); it is not part of upstream.

Measured with upstream's serving bench `v41bench_tf.py` (see tools/dsv41-hip/RESULTS.md; prompt set v1, `--levels 1,2,3,4,5,6,7,8 --prefill ''`, temperature 0, thinking off,
replies capped at 200 tokens and 150 for summary), total tok/s, the mean of the 8 category medians over 3 passes:

| Chats at once | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP4, 4.0 bpw, rdma_proxy, drafts on | 63.58 | 88.75 | 102.18 | 109.46 | 113.88 | 117.29 | 117.31 | 122.63 |
| Pair, 2.9 bpw (for comparison) | 48.55 | 63.96 | 73.73 | 77.06 | 80.40 | 83.23 | 88.08 | 91.80 |

One chat with a cold prompt read 1,330 to 1,490 tok/s from 8K to 128K tokens of context and wrote 77 to 85 tok/s with
drafts on (36 to 37 with drafts off). The boxes: four Ryzen AI Max+ 395 with 128 GB each, one ConnectX-6 Dx port each
on one 100 GbE switch (RoCE v2).

What the branch carries on top of upstream `93d28458`:

- The 2D split built into `native-hip`, sparse attention at 16 heads a rank, and a node's prompt experts on TP2's
  kernel path (TP4 and the pair gave the same prompt-logits sha256 on 8 frozen prompts at 2.9 bpw).
- TP4 rank files: `tf-dsv41-rank-reslice` and `tools/dsv41-hip/reslice_tp4.sh` cut them from the TP2 rank files.
- Four-rank collectives: RCCL through `Comm.coll`, and `rdma_proxy` at four ranks (one QP a peer), with prompt
  gathers pipelined over lanes (`TF_COLL_LANES`), `catRank` as a pair exchange, one-row quarters gathered straight
  into place (`COLL_SWAP12`) and multi-row quarters assembled with two copies (`VGATHER`).
- Faster attention linears on a 2D node (`TP4_FILL`) and the prompt split overlapped at world 4 (`PREFILL_SPLIT`
  with `PREFILL_ZIP2`).
- `first_light.sh` and `serve_lib.sh` start, check and stop four boxes (`tp4.conf`); `tfp-node0` is the word-checked
  remote end for the other boxes, usable as an ssh forced command; `tp4_orch_test.sh` checks the orchestration on
  stubs.
- Server and engine changes that also apply to a pair, several of them open upstream pull requests: the request
  queue cap (429), an idle connection's buffers given back, the collective timer (`TF_DS_COLL_TIME`), control-link
  keepalives, `_exit` after teardown on HIP, the `/cache` transfer routes (off by default), the picture cache
  (`ENCODE_CACHE_MIB`), the disk tier at world 4, a 256 MiB request body limit, and Pillow from the deploy
  (`TF_PIL_PATH`) so JPEG and other non-PNG pictures decode on boxes without Pillow.
- Edge-REAP routing statistics, passive and off by default (`REAP_STATS=1` on every rank): per expert and MoE
  layer, token counts, gate mass and the sampled norm of each routed expert's output, collected from normal traffic.
  `REAP_EVERY` (decode sample period), `REAP_PREFILL_ROWS` (prompt rows sampled per chunk), `REAP_BUCKETS` (domain
  cap), `REAP_DIR` and `REAP_SNAPSHOT_MIN` (periodic JSON snapshots) set it up. `GET /stats/reap` returns the
  export, in the format of `export_report` in `strix_moe_surgery.py`, behind the `CACHE_TRANSFER_KEY` bearer. A
  request picks its domain with `reap_domain` in the body or the `x-reap-domain` header. On the four boxes, replies
  were byte-identical with it on and off, and one chat decoded at 88.2 tok/s with it on against 88.4 off. Design and
  numbers: `docs/experiments/tp4-reap-stats.md`.
- A loop policy (`LOOP_POLICY` = `off`, `close`, `stop` or `count`; unset keeps `LOOP_GUARD`'s behaviour): an
  exact token cycle while thinking is open either ends the reply with `finish_reason` `"loop"` (`stop`), forces the
  think close (`close`), or is only reported (`count`). Every policy but `off` adds `tensorfold.loop` to the reply
  (on `/v1/chat/completions`, `/v1/messages` and `/v1/responses`) and counts `tensorfold:loop_detections_total` in
  `/metrics`.
- A guard for uneven splits at world 4: the engine refuses to open when the two pairs' exchanges would take different
  chunk counts for some tensor and row count, naming the tensor and the first row count that breaks, instead of
  deadlocking on the first such exchange.

To run it:

1. Build `native-hip` with a Triton kit that has the 16-head variants (`tools/dsv41-hip/tri/README.md`,
   `t_hipattn16.py`) and deploy the same build to all four boxes (`tools/dsv41-hip/BUILD_AND_RUN.md`). RCCL needs a
   gfx1151 `librccl` built against the engine's TheRock (`RCCL_LIB`).
2. Cut the four rank files from your TP2 bundle: `tools/dsv41-hip/reslice_tp4.sh TP2_BUNDLE OUT_BUNDLE 0 1 2 3`, and
   put rank g's file on box g. Each box also needs the Engram shards.
3. Copy `tools/dsv41-hip/tp4.conf` and fill in the lines marked SET (addresses, deploy, bundle). As shipped it is the
   first-light setup (RCCL, one stream, drafts off). The numbers above used these keys on top of it:
   `TRANSPORT=rdma`, `RDMA_DEV0`..`RDMA_DEV3` (each box's RoCE device), `COLL_SLOT_KIB=8704`,
   `COLL_PROXY_STRIPE_KIB=0`, `COLL_SWAP12=1`, `VGATHER=1`, `TP4_FILL=1`, `DRAFTS=1`, `GRAPHS=1`,
   `STREAMS=16`, `PARALLEL=16`, `CONTEXT=524288`, `POOL=22000000`, `MAX_TOKENS=524288`, `PREFILL_SPLIT=256`,
   `PREFILL_ZIP2=1`, `PROMPT_CACHE_GIB=4`, `LOGITS_SHA=0`, `AUX=`, `PEAK_GIB=88`.
4. On rank 1's box: `TF_SERVE_CONF=$HOME/tfp/run/conf/tp4.conf tools/dsv41-hip/first_light.sh plan` prints every
   rank's launch line; `... first_light.sh up` starts the four ranks (followers first, all GPU locks or none) and
   `... down` stops them.

## Known limits

- Tested on this exact setup only. Other Strix Halo machines, ROCm versions and cables may behave differently.
- Images are off unless `MODALITIES=text_image` is set. A request then takes at most 32 images (`MAX_IMAGES`),
  134,217,728 decoded pixels and 4,096 image tokens in all; past any of them it is refused with a 400.
- Provided as-is. Issues are welcome; support is best effort.

## Community

Setup help, benchmarks, model recipes and two-box tuning are on the [StrixFold Discord](https://discord.gg/D4FgAJnKd9).
Bug reports are welcome there or as GitHub issues. Measured StrixFold on your own boxes? Send the numbers with the
**Results from your hardware** issue form and they go on [docs/COMMUNITY-RESULTS.md](docs/COMMUNITY-RESULTS.md).
News and releases are also posted on X as [@Shoestring_Dev](https://x.com/Shoestring_Dev).

## Credits and licence

Apache-2.0, as upstream. TensorFold is by Ash Hart and the TensorFold contributors. The DeepSeek-V4.1 family this
builds on is by Bertholomus AI (Albert Lee). The EXL3 weight format follows ExLlamaV3 (MIT), credited in the kernel
files. Upstream's LICENSE and notices are unchanged; this fork's changes are listed in [CHANGES.md](CHANGES.md) and
the NOTICE file. Upstream's own README is kept at [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md).
