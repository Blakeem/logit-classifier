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
Two positions carry a fixed meaning instead. The escape label always means none of these,
and a score's letters always name the same rungs. Their mean mass follows the traffic, so
correcting them moves one request's answer with unrelated requests.

Banking77, 462 held out rows, Qwen3-VL-4B-Instruct at its fitted temperature.

| Prior | Accuracy | ECE |
|---|---|---|
| off | 0.606 | 0.083 |
| escape corrected, score shares the choice bucket | 0.621 | 0.114 |
| escape left alone, score in its own bucket | **0.630** | 0.103 |

The fix does not regress calibration. The accuracy gain is four rows in 462, near the noise
of this sample. The prior still costs calibration against no prior, because the shipped
temperature was fitted with the prior off. `banking77.py`, with and without `--no-prior`.

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
| `allow_bf16_reduced_precision_reduction` | **0.5** | **0.75** |
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
could not reach it. `sdpa_kernel` turns the math backend **off** while it holds the CUDA
backends, so this setting is live only when `_usable_attention_backends` returns nothing and
no `sdpa_kernel` block is entered. ComfyUI turns it **on** at import, at
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

| Request shape | ms | peak GB | suffix tokens | chunks | |
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

3.29x and 5.06x on the two shapes that mix widths, each giving back about 3 GB. Nothing
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
| 512 | 610.9 ms, 9.91 GB, 5 chunks | 2015.7 ms, 5 chunks |
| 1024 | 608.3 ms, 10.47 GB, 3 chunks | 2014.9 ms, 5 chunks |
| **2048** | **626.0 ms, 10.66 GB, 3 chunks** | **2015.8 ms, 5 chunks** |
| 4096 | 793.5 ms, 11.03 GB, 2 chunks | 2286.5 ms, 4 chunks |
| 8192 | 1119.4 ms, 11.84 GB, 2 chunks | 2289.5 ms, 4 chunks |
| none | 2062.2 ms, 13.66 GB, 1 chunk | 2299.1 ms, 4 chunks |

Anything from 512 to 2048 measured the same on both, and 4096 upward is clearly worse.
The value is not a knife edge. 2048 sits in the middle of the flat band and leaves the
most room before two near-equal wide branches are split apart.

`ab_branch_packing.py`.

## Throughput and where it falls off

Warm model, RTX 3090 Ti. Adding questions to a record is close to free up to about 8,
because the fixed cost of two forward passes dominates a short record.

| Questions per record | Total ms | ms per tag | Records per hour |
|---|---|---|---|
| 1 | 138 | 138 | 26,100 |
| 4 | 136 | 34 | 26,500 |
| 8 | 140 | 17.5 | 25,800 |
| 32 | 278 | 8.7 | 13,000 |

The state is prefilled once per record, so a longer record costs more.

| State tokens | Total ms, 8 questions | ms per tag | Records per hour |
|---|---|---|---|
| 172 | 140 | 17.5 | 25,700 |
| 556 | 185 | 23 | 19,500 |
| 2,092 | 451 | 56 | 8,000 |
| 8,236 | 1,575 | 197 | 2,300 |
| 32,812 | 41,515 | 5,189 | 87 |

Prefill runs at 46 to 48 TFLOP/s from about 2,000 tokens up, near this card's practical
BF16 ceiling. Below that the time is almost all kernel launch rather than arithmetic, so a
428 token request spends 0.14 ms on the math. The last row falls off the ceiling for the
reason under Known weaknesses. `benchmark.py`.

## The attention backend has to be pinned on Windows

Torch 2.11 ships no FlashAttention kernel for Windows. Left to choose for itself the
dispatcher reaches the math backend, which builds the full attention matrix. Measured on an
8,192 token prefill.

| Backend selection | Time | Peak memory |
|---|---|---|
| left to the dispatcher | 147,221 ms | 27.6 GB |
| pinned to cudnn and efficient | 1,291 ms | 9.2 GB |

`HFBackend` probes the backends at startup, at the model's own dtype, and pins the working
ones. `benchmark.py`.

## Bugs found and fixed

| Bug | Effect | Fix |
|---|---|---|
| score used the choice confidence formula | wrong confidence on every score answer | ported both official formulas |
| attention probe ran in bfloat16 whatever the model dtype | any other dtype crashed on the first forward | probe at the model's dtype |
| attention probe used one call shape | approved a backend that then failed | probe causal and masked, with grouped query attention |
| abstain combined groups by product | shrinks with group count, never fires at scale | weakest decline |
| image source detected by string length | an 8x8 PNG encodes shorter than the threshold and was read as a path | check whether the file exists |
| M-RoPE offset dropped on branch rows | Qwen3-VL raises rather than misplacing tokens | read `rope_deltas` after the prefix pass |
| prior learned and corrected the escape label | traffic that often declines taught the prior to undo a decline | prior over the lettered options only |
| a joint score shared a prior bucket with a choice of the same width | a score's top rung learned the choice traffic's escape mass | score has its own bucket |

## Known weaknesses

**The split path.** Three defects share one cause, that groups above 52 options cannot see
each other. Group assignment bias costs 13.9 points, temperature leaks into accuracy, and
the abstain score falls from 0.878 to 0.639. Parked by choice. Candidates are OpenJev's
chunk winners tournament and a second pass over each group's winner. Neither is measured.

**Noul temperature is unfitted.** The shipped 1.25 was fitted on choice questions. Booleans
now saturate near 0 and 1, so the best threshold sits at 0.9999 rather than near 0.5.

**Noul regressed 8 points** moving to the vision model, and image tagging leans on noul.

**Very long states fall off the prefill ceiling.** At 33,068 prefilled tokens the card runs
at 7.1 TFLOP/s against 47.9 at 8,492. The prefix key value cache is about 4.9 GB at that
length and every chunk copies it, which leaves a 24 GB card no room. Expanding the cache
once and cropping between chunks was measured and is worse, at 12,732 ms against 1,710 ms
and 23.66 GB against 15.12 GB, because a cropped cache is a non-contiguous view. Unsolved.
`benchmark.py`.

## Layout

`tests/` holds the pytest suite, 146 unit tests and 14 that need the GPU. `tests-AB/` holds
the measurement harnesses, matching ComfyUI-ContextAnchoredTileRefine. Fixtures live in
`tests/fixtures/` and `tests-AB/ab_env.py` points at them.
