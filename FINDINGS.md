# Findings

What changed, why, and what was measured. Every number here came from a script in
`tests-AB/`, on an RTX 3090 Ti. Reproduce any row by running the script named beside it.

## The pipeline every finding attaches to

```
request JSON
  → parse        validate, lift any image out of the state
  → label        give each option one letter, A to Z then a to z
  → assemble     build the shared prefix once, one suffix per branch
  → prefill      append "Answer: (" to the rendered chat template
  → score        encode the prefix once, broadcast its cache, one batched forward
  → read         gather the logits at the candidate letter ids
  → calibrate    subtract the learned label prior, divide by the fitted temperature
  → normalize    softmax over those letters only, in float64
  → map          letters back to option names
response JSON
```

A branch is one prompt whose last token position carries a distribution. A question needs
one branch, or several when it has levels or more than 52 options.

## Model

| | Qwen3-4B-Instruct-2507 | Qwen3-VL-4B-Instruct |
|---|---|---|
| Banking77, 462 held out rows | 0.554 | **0.626** |
| eval set, 39 items | 0.897 | **0.949** |
| eval set, score questions | 0.750 | **1.000** |
| eval set, noul questions | **1.000** | 0.917 |
| median latency | 171 ms | **169 ms** |
| reads images | no | **yes** |

The vision model is better at text as well as capable of images, at the same speed, so it
is the default. `ab_temperature.py`, `evaluate.py`.

Noul is the one regression, down 8 points, and noul is what image tagging uses most.

## Temperature belongs to the model

Fitted on the held out half of a 924 row Banking77 sample, chosen on calibration error.

| Model | fitted T | ECE at its own T | ECE at the other model's T |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | 1.25 | 0.088 | 0.565 |
| Qwen3-4B-Instruct-2507 | 6.0 | 0.029 | 0.088 |

Nearly a 5x difference between two models of the same size and family, so a single constant
was wrong. `config.py` holds `FITTED_TEMPERATURES`, and an unfitted model gets 2.5.

The fit follows calibration error, not accuracy. Accuracy climbs with temperature above 52
options, but that is the split path defect below, not a gain. `ab_temperature.py`.

## The prior corrects letters, not fixed positions

The running prior subtracts the model's standing preference for one letter over another.
Some positions carry a fixed meaning instead. The escape label always means none of these,
a score's letters always name the same rungs, and a noul's `A` is always yes. Their mean mass
follows the traffic, so correcting them moves one request's answer with unrelated requests.
The prior therefore covers the lettered options of choice and member branches only.

Banking77, 462 held out rows, Qwen3-VL-4B-Instruct at its fitted temperature.

| Prior | Accuracy | ECE |
|---|---|---|
| off | 0.606 | 0.083 |
| escape corrected, score shares the choice bucket | 0.621 | 0.114 |
| escape left alone, score in its own bucket | **0.630** | 0.103 |

The fix does not regress calibration. The accuracy gain is four rows in 462, near the noise
of this sample. The prior still costs calibration against no prior, because the shipped
temperature was fitted with the prior off. `banking77.py`, with and without `--no-prior`.

A noul under the prior shows the same coupling. Banking77 statements ("This message is about
<intent>", true for the row's own intent) ran as request streams at four true rates, the first
32 requests of each stream unscored.

| Stream | Prior | AUC | Recall at 0.5 | False positives at 0.5 | Recall at 0.9999 |
|---|---|---|---|---|---|
| 50% true | off | 0.9538 | 0.891 | 0.065 | 0.050 |
| 50% true | on | 0.9531 | 0.891 | 0.065 | 0.059 |
| 80% true | off | 0.9419 | 0.872 | 0.091 | 0.055 |
| 80% true | on | 0.9410 | 0.848 | 0.068 | 0.006 |
| 95% true | off | 0.9428 | 0.878 | 0.083 | 0.056 |
| 95% true | on | 0.9409 | 0.821 | 0.000 | 0.005 |
| 20% true | off | 0.9439 | 0.878 | 0.078 | 0.000 |
| 20% true | on | 0.9435 | 0.878 | 0.102 | 0.122 |

The AUC does not move, since the correction is one shift of the yes/no gap. The operating point
moves with the traffic instead. A request took 126.6 ms with the prior off and 126.4 ms with it
on, so the prior has no speed cost or gain. `ab_prior_traffic.py`.

ContextAnchoredTileRefine's tags bench kept 8.2 correct tags per tile with the prior off and
5.4 with it on, over 45 judged tiles on 15 pictures. The bench ran on 00518c6, before 3ac0294
took the prior off nouls, so its prior also corrected the verify and locate nouls. The loss grew
over the later tiles, which the earlier tiles' mostly true verify answers had trained.

| Tile in run order | Correct tags, prior off | Correct tags, prior on |
|---|---|---|
| first | 114 | 111 |
| second | 110 | 80 |
| third | 146 | 52 |

The first tile kept the same count on 13 of 15 pictures. It lost 2 tags on dragon4k and 1 on
market4k, both under the long prompt. `ab_tags_bench.py` in
ContextAnchoredTileRefine's `tests-AB/`, arms `prior-off` and `prior-on`, added in its commit
bf6c7ba.

## Option order helps above 52 options and nowhere else

| Options | Letterings | Accuracy | ECE |
|---|---|---|---|
| 10, one branch | 1 | 0.881 | 0.049 |
| 10, one branch | 4 | 0.881 | 0.025 |
| 77, split | 1 | 0.554 | 0.097 |
| 77, split | 4 | **0.693** | 0.288 |

OpenJev ships the same idea as `READOUT_PERMS`, default off, with a code comment claiming
about 16 points. The gain reproduces, but it is group assignment rather than letter
position. Ten options fit one branch, where relettering changes nothing measurable.

Shipped as `LOGIT_PERMUTATIONS`, default 1. Calibration gets worse as accuracy improves, and
no temperature between 0.5 and 50 repaired it.

## Abstain

Every choice question carries a `none of these` label, reported as `abstain` outside
`probabilities`. Measured against rows whose correct label was deleted from the list, so
declining was the only right answer.

| Setup | AUC |
|---|---|
| one branch, 10 options | **0.878** |
| split path, 77 options | 0.639 |

The split path is weak because a group that does not hold the answer declines whether or not
the answer sits in another group.

Two alternatives were tested and rejected. Mass on letters nobody offered reaches AUC 0.806
alone, but adding it to the escape label drops the pair to 0.850. Combining across groups by
product scored 0.616 against 0.639 for the weakest decline, and its scale shrinks with the
group count, so five honest half declines would read as 0.03. The weakest decline ships.

Offering the label does not measurably change accuracy. It moved by two to three examples in
462, inside the noise of that sample. `ab_abstain` measurements, `ab_temperature.py`.

## Abstain and confidence answer different questions

| What to catch | Signal | AUC |
|---|---|---|
| nothing in the list fits | `abstain` | 0.878 |
| the tag is probably wrong | `confidence` | 0.864 |

They correlate at -0.64, so they overlap, but each wins its own job. A tagging pipeline
should check both.

## Several true answers need one noul each

| Method | AUC across tiles | AUC within a tile | best F1 |
|---|---|---|---|
| one `noul` per fragment | **0.963** | **1.000** | **0.909** |
| one `choice` over all fragments | 0.667 | 0.685 | 0.516 |

A choice is normalized and sums to 1. On a tile holding four fragments it gave the winner
1.00 and the other three 0.00. The published result that symbol scoring beats independent
scoring by 9.7 points covers single label tasks, and does not transfer here.
`ab_multi_label.py`.

## The wording of a boolean does not matter

Yes and no against true and false, correct and incorrect, present and absent, visible and
hidden, agree and disagree. AUC spanned 0.946 to 0.957 on 462 balanced text statements and
0.960 to 0.969 on the tiles. That is inside noise at these sample sizes.

The model writes "no" more readily than "yes" when generating. The readout never asks it to
write anything, so both words are equally reachable at the one position we read.
`ab_noul_wording.py`.

## Confidence formulas

TypeSafe publishes two, and we were using the choice one for score questions. A score is
ordinal, so `[0, 0.5, 0.5, 0, 0]` and `[0.5, 0, 0, 0, 0.5]` are different answers that the
peak height formula scores identically at 0.375. The official formula gives 0.583 and 0.0.

Verified by reading `system_one_adapter._utils.confidence_metrics` from TypeSafe's own PyPI
package, not by trusting a third party's description of it.

## Which torch globals decide bit-exactness

`backends/hf.py` holds a set of process-global torch settings around each forward pass. On
the CUDA attention path, only one of them changes a logit on either shipped model.

| setting flipped away from pinned | Qwen3-VL-4B-Instruct | Qwen3-4B-Instruct-2507 |
|---|---|---|
| `cudnn.benchmark` | 0.0 | 0.0 |
| `cudnn.deterministic` | 0.0 | 0.0 |
| `allow_bf16_reduced_precision_reduction` | **0.375** | **1.125** |
| float32 matmul precision | 0.0 | 0.0 |

Max absolute logit shift against the fully pinned run, window removed so the process
globals reach the forward pass. The forward runs in bfloat16, so the fp32 matmul setting
governs matmuls this graph does not have, and the cudnn pair governs convolutions that
only the vision tower's patch embed uses. The three inert settings are kept because
`cudnn.benchmark` picks convolution algorithms by timing, which is specific to the card
and the model, and this was measured on one card with two models.

Scoping the settings to the forward pass moved nothing. The same request returns the same
logits, and a run under a hostile host state matches a run under a pinned one, both at a
gap of 0.0 on both models.

`allow_fp16_bf16_reduction_math_sdp` is held for a different reason and the sweep above
could not reach it. The math backend is **off** while `sdpa_kernel` holds the backends the
probe found. When `_usable_attention_backends` returns nothing, the `FALLBACK_ATTENTION` block
puts math last. So this setting is live only on a pass that flash, cuDNN and memory efficient
attention all reject. ComfyUI turns it **on** at import, at
`comfy/model_management.py:569`, so inside ComfyUI that path would run with reduced
precision reductions.

Forcing the math backend measures it on the CPU, with no card or weights.

| bfloat16 attention, math backend | bitwise equal | max abs diff |
|---|---|---|
| `(1, 4, 256, 128)` | no | 0.00977 |
| `(1, 4, 1024, 128)` | no | 0.00391 |
| `(8, 4, 256, 128)` | no | 0.00781 |

That is far above a low bit, so the math path needs the setting held.
`ab_math_sdp_reduction.py`.

Three more settings were considered and left to the host. `cudnn.enabled` is off on hosts
where cuDNN does not work, so forcing it on would break the machine that turned it off.
`cudnn.conv.fp32_precision` and `cudnn.allow_tf32` govern fp32 convolutions, and the
forward runs in bfloat16. `torch.use_deterministic_algorithms` moves toward determinism, so
a host that sets it costs us nothing.

When the probe finds no usable kernel the pass pins flash, cudnn, mem efficient and math in
that order instead of leaving the choice to the host's enable flags. Without that pin, two
hosts could dispatch one request to two different kernels.

`set_float32_matmul_precision` writes the cuda and the mkldnn matmul slots together, and
its getter reports both `none` and `ieee` as `highest`. So restoring through that setter
alone leaves the mkldnn slot changed, which is why both slots are saved raw.

`ab_determinism_scope.py`.

## Packing branches by suffix length

Every row in a chunk is left-padded to that chunk's longest suffix, and each padded
token costs a full forward plus attention over the whole prefix. Taking branches in the
order the caller sent them drags short branches to the longest width.

| Request shape | ms | peak GiB | suffix tokens | chunks | |
|---|---|---|---|---|---|
| mixed, four choices plus 12 nouls plus 4 scores | 2055.6 | 13.66 | 14600 | 1 | request order |
| | **623.9** | **10.66** | **2614** | 3 | packed |
| one 45 option choice beside 20 nouls | 2152.9 | 13.92 | 15330 | 1 | request order |
| | **425.2** | **10.94** | **1310** | 2 | packed |
| 256 nouls at the Jev cap | 1370 | 8.75 | 7584 | 8 | both, bitwise identical |
| one 77 option choice, split | 272 | 8.61 | 1282 | 1 | both, bitwise identical |
| 24 tile fragments | 198 | 8.63 | 696 | 1 | both, bitwise identical |
| 30 three option choices | 396 | 8.99 | 2160 | 1 | both, bitwise identical |
| 12 twenty option choices | 586 | 9.24 | 3960 | 1 | both, bitwise identical |

3.29x and 5.06x on the two shapes that mix widths, each giving back about 3 GiB. Nothing
at all on the five that do not, which is the point. The sort is stable, so equal lengths
keep the caller's order and the grouping stays a pure function of the request.

The two mixed shapes move low bits, by at most 0.375 and 0.5. Changing the padded width
or the batch composition of a bf16 forward does that. The shapes whose bits move are the
shapes that were three to five times too slow.

**The token ceiling is checked only when a branch widens the chunk it joins.** A request
whose branches are all one width has no padding to remove, so the ceiling would only add
chunks. Checking it unconditionally re-chunked any uniform request above 64 tokens of
width, which is every choice question, and moved its logits for no gain.

| ceiling | mixed shape, 800 token prefix | mixed widths, long prefix |
|---|---|---|
| 512 | 610.9 ms, 9.91 GiB, 5 chunks | 2015.7 ms, 5 chunks |
| 1024 | 608.3 ms, 10.47 GiB, 3 chunks | 2014.9 ms, 5 chunks |
| **2048** | **626.0 ms, 10.66 GiB, 3 chunks** | **2015.8 ms, 5 chunks** |
| 4096 | 793.5 ms, 11.03 GiB, 2 chunks | 2286.5 ms, 4 chunks |
| 8192 | 1119.4 ms, 11.84 GiB, 2 chunks | 2289.5 ms, 4 chunks |
| none | 2062.2 ms, 13.66 GiB, 1 chunk | 2299.1 ms, 4 chunks |

Anything from 512 to 2048 measured the same on both, and 4096 upward is clearly worse.
The value is not a knife edge. 2048 sits in the middle of the flat band and leaves the
most room before two near-equal wide branches are split apart.

`ab_branch_packing.py`.

## Throughput and where it falls off

Warm model, RTX 3090 Ti. Each row repeats one request, so its sequence lengths are already
seen. Adding questions to a record is close to free up to about 8, because the fixed cost of
two forward passes dominates a short record.

| Questions per record | Total ms | ms per tag | Records per hour |
|---|---|---|---|
| 1 | 135 | 135 | 26,700 |
| 4 | 132 | 33 | 27,300 |
| 8 | 135 | 16.9 | 26,600 |
| 32 | 295 | 9.2 | 12,200 |

The state is prefilled once per record, so a longer record costs more.

| State tokens | Total ms, 8 questions | ms per tag | Records per hour | Peak GiB |
|---|---|---|---|---|
| 172 | 144 | 18.0 | 24,900 | 8.36 |
| 556 | 199 | 24.9 | 18,100 | 8.48 |
| 2,092 | 494 | 62 | 7,300 | 8.92 |
| 8,236 | 1,805 | 226 | 2,000 | 10.23 |
| 32,812 | 11,453 | 1,432 | 314 | 15.23 |

The TFLOP/s count covers the weights' FLOPs only. Prefill runs at about 42 TFLOP/s from 2,348
to 8,492 prefilled tokens. A 428 token prefill runs at 26.3 TFLOP/s and a 33,068 token prefill
at 25.6. Attention work grows with the square of the state length, so the count leaves out the
most work on the 33,068 token row. The peak is `torch.cuda.max_memory_allocated` in GiB, weights
included. `benchmark.py`.

## The memory efficient kernel, not cuDNN

Torch 2.11 ships no FlashAttention kernel for Windows, so `HFBackend` probes the backends at
startup, at the model's own dtype, and pins one. cuDNN builds an execution plan the first time it sees a sequence length, and served requests
rarely repeat a length. The memory efficient kernel has no such cost, but it rejects grouped
key and value heads. transformers passes grouped heads whenever a pass has no mask, which is
the prefix pass. So the model loads under its own attention name, which expands the heads
before the kernel sees them.

Both arms ran on one loaded model.

| | cuDNN, grouped heads | memory efficient, expanded heads |
|---|---|---|
| 40 requests of unseen length, 96 to 2,865 tokens, p50 | 1,501 ms | 335 ms |
| the same 40 requests repeated, p50 | 335 ms | 327 ms |
| image request of unseen length, 132 image tokens | 2,257 ms | 214 ms |
| 8,192 token state and 8 questions, unseen | 3,664 ms | 1,927 ms |
| 8,192 token state and 8 questions, repeated | 1,672 ms | 1,742 ms |
| peak memory at 8,192 tokens | 15.76 GiB | 15.76 GiB |
| Banking77, 462 held out rows, accuracy | 0.6255 | 0.6212 |
| Banking77, 462 held out rows, ECE at T=1.25 | 0.0882 | 0.0834 |

The two kernels round differently in bfloat16. Logits moved by a median of 0.25 and at most
1.5, and the top letter changed on 36 of 924 rows. Accuracy moved by 2 of 462 held out rows,
which is inside the noise of that sample, so the fitted temperature stands. The same request
still returns bitwise equal logits on 924 of 924 rows. `ab_attention_kernel.py`.

## One frozen prefix cache, not a copy per chunk

Each chunk used to deep copy the prefix cache and broadcast it across its rows, which stored
rows times the prefix in every layer for the whole forward. The prefix cache is now frozen
after the prefix pass, and each layer joins the broadcast prefix to the chunk's own keys and
values only for that layer's attention.

Both arms ran on one idle GPU, the commit before the change against the commit after.

| Request | peak before | peak after | logits |
|---|---|---|---|
| mixed, four choices plus nouls plus scores | 10.67 GiB | 8.75 GiB | bitwise identical |
| skewed, one wide choice beside twenty nouls | 10.95 GiB | 8.76 GiB | bitwise identical |
| uniform, 256 nouls at the Jev cap | 8.74 GiB | 8.39 GiB | bitwise identical |
| split, one 77 option choice | 8.61 GiB | 8.40 GiB | bitwise identical |
| tile fragments, 24 nouls | 8.63 GiB | 8.36 GiB | bitwise identical |
| uniform, 30 three option choices | 8.99 GiB | 8.47 GiB | bitwise identical |
| uniform, 12 twenty option choices | 9.24 GiB | 8.60 GiB | bitwise identical |
| 8,192 token state and 8 questions | 15.76 GiB | 10.24 GiB | bitwise identical |
| image state, top-L.png | 8.37 GiB | 8.34 GiB | bitwise identical |

The peaks in the packing and kernel sections above were measured with the copy.
`ab_frozen_prefix_cache.py`.

A request of 33,068 prefilled tokens took 155,903 ms at 1.9 TFLOP/s on the memory efficient
kernel and 52,813 ms on cuDNN while each chunk copied the prefix cache. The frozen cache brings
it to 11,453 ms. `benchmark.py`.

That request's prefix cache holds about 4.5 GiB. The figure is derived from the shipped config
rather than measured, as 147,456 bytes per token (36 layers × 8 key value heads × 128 head_dim
× 2 tensors × 2 bytes) times the 32,812 prefix tokens.

## One presence wording, two threshold pairs

Correct tags per tile at a pooled precision, over 905 judged tile candidates on 23 tiles,
from the Logit Tagger's `tests-AB/ab_presence_wording.py` (round 4).

| Wording | P 0.95 | P 0.97 | P 0.98 | P 0.99 |
|---|---|---|---|---|
| "This image visibly contains {TAG}" | 17.43 | 12.78 | 9.87 | 6.04 |
| "Is there a {TAG} in this image?" (0.3.0, article by rule) | 17.35 | 11.30 | 2.39 | 2.09 |
| "Present in this image: {TAG}" | 17.61 | 8.74 | 8.13 | 6.74 |

The shipped fp8 tile pass cut offline, per tile and prompt over 69 runs, from
ContextAnchoredTileRefine's `tests-AB/ab_tags_paired.py`.

| Thresholds | Correct | Wrong or vague | Precision |
|---|---|---|---|
| broad (0.999, 0.9999) | 12.32 | 0.54 | 0.958 |
| strict (0.99998, 0.99998) | 7.33 | 0.17 | 0.977 |

Every paired interval against "This image visibly contains {TAG}" at 0.95 and 0.98 spans zero.
It needs no grammar, and the tile pass already asked it, so its thresholds carry over. On
declared COCO tag lists "Is there a {TAG} in this image?" (0.3.0, article by rule) kept more
correct tags at precision 0.98 (3.79 against 1.90 per image), which is the cost of one
wording.

## Bugs found and fixed

| Bug | Effect | Fix |
|---|---|---|
| score used the choice confidence formula | wrong confidence on every score answer | ported both official formulas |
| attention probe ran in bfloat16 whatever the model dtype | any other dtype crashed on the first forward | probe at the model's dtype |
| attention probe used one call shape | approved a backend that then failed | probe the causal and masked forms the forward sends |
| cuDNN attention built an execution plan for every new sequence length | about a second on almost every served request | memory efficient kernel, with grouped heads expanded before it |
| abstain combined groups by product | shrinks with group count, never fires at scale | weakest decline |
| image source detected by string length | an 8x8 PNG encodes shorter than the threshold and was read as a path | check whether the file exists |
| M-RoPE offset dropped on branch rows | Qwen3-VL raises rather than misplacing tokens | read `rope_deltas` after the prefix pass |
| prior learned and corrected the escape label | traffic that often declines taught the prior to undo a decline | prior over the lettered options only |
| a joint score shared a prior bucket with a choice of the same width | a score's top rung learned the choice traffic's escape mass | no prior on a score |
| the yes/no and score layouts learned the traffic's answer rate | mostly true traffic cut recall at 0.9999 from 0.055 to 0.006 | prior over choice and member branches only |

## Known weaknesses

**The split path.** Three defects share one cause, that groups above 52 options cannot see
each other. Group assignment bias costs 13.9 points, temperature leaks into accuracy, and
the abstain score falls from 0.878 to 0.639. Parked by choice. Candidates are OpenJev's
chunk winners tournament and a second pass over each group's winner. Neither is measured.

**Noul temperature is unfitted.** The shipped 1.25 was fitted on choice questions. Booleans
now saturate near 0 and 1, so the best threshold sits at 0.9999 rather than near 0.5.

**Noul regressed 8 points** moving to the vision model, and image tagging leans on noul.

## Layout

`tests/` holds the pytest suite, the unit tests and 14 tests that need the GPU. `tests-AB/` holds
the measurement harnesses, matching ComfyUI-ContextAnchoredTileRefine. Fixtures live in
`tests/fixtures/` and `tests-AB/ab_env.py` points at them.
