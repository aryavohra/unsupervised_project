# Experiment Log Audit

Audit date: 2026-05-26.

Scope: local experiment logs under `results/` and `novita*_logs/`, plus the local runner scripts that explain some console-only logs.

## Summary

Most canonical logs are self-describing: they include printed run config such as `Max train steps`, `XSA mode`, gate flags, validation controls, and a final validation or training/memory result. Console-only logs usually contain the result but not enough configuration by themselves; they are only clear when paired with their canonical `.txt` log or runner script.

The main gap is the two XSA interpretation logs: they record config, but no clear saved artifact path, metric summary, or result line is present locally.

## Full-Run / Target Logs

| Log group | Configuration clarity | Result clarity | Result |
|---|---|---|---|
| `results/omd-no-attn-gate-stop328-1450-20260525_225601/canonical.log` | Good: `output-metric-diag`, attention gate disabled, `max_train_steps=1450`, `val_every_after_step=1385`, `stop_val_loss_below=3.28` | Good | stopped at step `1413`, val `3.2799`, `651.294s`, peak/reserved `37327/52430 MiB` |
| `results/omd-no-attn-gate-stop328-1450-20260525_225601/train.log` | Result-only: use paired `canonical.log` for exact config | Good | same as above |
| `results/laplacian-no-attn-gate-stop328-1450-20260525_234745/canonical.log` | Good: `laplacian`, attention gate disabled, `max_train_steps=1450`, `val_every_after_step=1385`, `stop_val_loss_below=3.28` | Partial: final validation is clear, but checkpoint save failed after validation because the remote filesystem was full | reached step `1450`, val `3.3001`, `658.988s`, checkpoint incomplete/ignored |
| `results/laplacian-no-attn-gate-stop328-1450-20260525_234745/train.log` | Result-only: use paired `canonical.log` for exact config | Partial: includes final validation and checkpoint I/O failure | same as above |
| `novita_record_detached_no_attn_gate_stop328_1450_logs/record-detached-no-attn-gate-stop328-1450.txt` | Good: `record`, `XSA detach projection=True`, attention gate disabled, `max_train_steps=1450`, `val_every_after_step=1385`, `stop_val_loss_below=3.28` | Partial: final validation is clear, but checkpoint save failed after validation because the remote filesystem was full | stopped at step `1450`, val `3.2789`, `665.314s`, checkpoint incomplete/ignored |
| `novita_record_detached_no_attn_gate_stop328_1450_logs/record-detached-no-attn-gate-stop328-1450.console.log` | Result-only: use paired `.txt` for exact config | Partial: includes final validation and checkpoint I/O failure | same as above |
| `results/novita_xsa_no_gate_target328/codex-xsa-no-gate-target328.txt` | Good: `record`, attention gate disabled, `max_train_steps=2500`, `val_every_after_step=1385`, `stop_val_loss_below=3.28` | Good | stopped at step `1404`, val `3.2799`, `638.155s`, peak/reserved `37328/52408 MiB` |
| `results/novita_xsa_no_gate_target328/console.log` | Result-only: use paired canonical `.txt` and `launch.log` | Good | same as above |
| `results/novita_xsa_no_gate_target328/launch.log` | Partial launch metadata only | No result | paired with `codex-xsa-no-gate-target328.txt` |

## 100-Step / 500-Step Ablations

| Log group | Configuration clarity | Result clarity | Result |
|---|---|---|---|
| `novita_laplacian_ablation_logs/laplacian-ablation-record-100.txt` | Good | Good | val `5.0153`, `1221.65ms/step`, peak/reserved `31408/39090 MiB` |
| `novita_laplacian_ablation_logs/laplacian-ablation-laplacian-100.txt` | Good | Good | val `5.2321`, `1138.61ms/step`, peak/reserved `31408/39088 MiB` |
| `novita_laplacian_ablation_logs/laplacian-ablation-laplacian-gated-100.txt` | Good | Good | val `5.2090`, `1166.06ms/step`, peak/reserved `31408/39088 MiB` |
| `novita_laplacian_ablation_logs/laplacian-ablation-suite-100.console.log` | Recoverable from `run_laplacian_ablation_100.sh` and paired `.txt` logs | Good aggregate console output | final line belongs to `laplacian-gated`: val `5.2090` |
| `novita_substoch_ablation_logs/substoch-ablation-record-100.txt` | Good | Good | val `4.9960`, `1229.08ms/step`, peak/reserved `31408/39090 MiB` |
| `novita_substoch_ablation_logs/substoch-ablation-substoch-100.txt` | Good | Good | val `5.0223`, `1144.26ms/step`, peak/reserved `31408/39090 MiB` |
| `novita_substoch_ablation_logs/substoch-ablation-substoch-delta-100.txt` | Good | Good | val `5.2326`, `1146.76ms/step`, peak/reserved `31408/39090 MiB` |
| `novita_substoch_ablation_logs/substoch-ablation-suite-100.console.log` | Recoverable from `run_substoch_ablation_100.sh` and paired `.txt` logs | Good aggregate console output | final line belongs to `substoch-delta`: val `5.2326` |
| `novita_substoch_500_logs/substoch-500-val100.txt` | Good | Good | val `4.1742`, `590.93ms/step`, peak/reserved `31408/39108 MiB` |
| `novita_substoch_500_logs/substoch-500-val100.console.log` | Result-only; paired `.txt` has config | Good | same as above |
| `novita_substoch_delta_val100_stop328_logs/substoch-delta-val100-stop328.txt` | Good: `substoch-delta`, `val_every_after_step=1385`, `stop_val_loss_below=3.28` | Good | final val `3.3128` at step `1385`, `457.86ms/step`, peak/reserved `37327/52150 MiB` |
| `novita_substoch_delta_val100_stop328_logs/substoch-delta-val100-stop328.console.log` | Result-only; paired `.txt` has config | Good | same as above |
| `novita_substoch_resume500_stop328_logs/substoch-resume500-stop328.txt` | Good but duplicated config because it appears to include resume/source context | Good | final val `3.2857` at step `1385`, `738.12ms/step`, peak/reserved `37327/52730 MiB` |
| `novita_substoch_resume500_stop328_logs/substoch-resume500-stop328.console.log` | Result-only; paired `.txt` has config | Good | same as above |

## 200-Step Paired-Head XSA / Gate Ablation

All four runs used `/root/xsa_gate_ablation/.venv` on Novita 1xH100 with PyTorch `2.10.0+cu126`, CUDA `12.6`, Triton `3.6.0`, fused CE enabled, `--xsa-mode record`, `--max-train-steps 200`, and `--val-loss-every 50`. The baseline files use record XSA only on non-paired attention layers. The paired-XSA files also apply record XSA to paired-head attention outputs.

| Log | Variant | Sparse attention gate | Val @50 | Val @100 | Val @150 | Final val @200 | Train time | Step avg | Peak/reserved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `novita_record_paired_comparison_200_logs/record-baseline-gate-on-200-val50-20260526.txt` | baseline record | enabled | `5.6741` | `5.0188` | `4.7409` | `4.5758` | `49.001s` | `245.00ms` | `37327/53158 MiB` |
| `novita_record_paired_comparison_200_logs/record-baseline-gate-off-200-val50-20260526.txt` | baseline record | disabled | `5.6301` | `5.0202` | `4.7571` | `4.5584` | `47.773s` | `238.87ms` | `37327/51848 MiB` |
| `novita_record_paired_comparison_200_logs/record-pairedxsa-gate-on-200-val50-20260526.txt` | paired-head record XSA | enabled | `5.6658` | `5.0133` | `4.7380` | `4.5650` | `49.006s` | `245.03ms` | `37327/53158 MiB` |
| `novita_record_paired_comparison_200_logs/record-pairedxsa-gate-off-200-val50-20260526.txt` | paired-head record XSA | disabled | `5.6181` | `5.0053` | `4.7168` | `4.5777` | `47.914s` | `239.57ms` | `37327/51848 MiB` |

Takeaways: disabling the sparse attention gate remains faster by about `5-6ms/step`. Paired-head XSA helps the gate-on run versus baseline gate-on (`4.5650` vs `4.5758`) at no meaningful runtime cost. With the gate disabled, paired-head XSA improves intermediate losses through step `150` but ends slightly worse at step `200` (`4.5777` vs `4.5584`), so it does not currently beat the no-gate baseline on this short fixed-step metric.

## Cross-Run Comparison

These are the runs most relevant to the current branch decision. They are not all perfectly matched: some stop at the `3.28` target, some run to a fixed terminal step, and two checkpoint writes failed after validation due the remote disk filling. The validation/timing numbers are still usable.

| Variant | Sparse attention gate | Stop/terminal step | End val | Train time | Step avg | Notes |
|---|---:|---:|---:|---:|---:|---|
| `record` | disabled | `1404` | `3.2799` | `638.155s` | `454.53ms` | Best target-hitting run in local logs. |
| `record` + detached XSA projection | disabled | `1450` | `3.2789` | `665.314s` | `458.84ms` | Hit target only at terminal step; checkpoint failed from disk-full. Not a win. |
| `output-metric-diag` | disabled | `1413` | `3.2799` | `651.294s` | `460.93ms` | Reaches same target loss later and slower than `record` no-gate. |
| `laplacian` | disabled | `1450` | `3.3001` | `658.988s` | `454.47ms` | Did not reach `3.28`; checkpoint failed from disk-full. |
| `substoch` resumed from 500-step checkpoint | enabled | `1385` | `3.2857` | `653.232s` | `738.12ms` | Did not reach target by default terminal step; resume timing not directly comparable. |
| `substoch-delta` | enabled | `1385` | `3.3128` | `634.129s` | `457.86ms` | Much worse end loss; not promising. |
| `substoch` 500-step run | enabled | `500` | `4.1742` | `295.467s` | `590.93ms` | Intermediate diagnostic only. |
| `substoch-delta` 100-step run | enabled | `100` | `5.2326` | `114.676s` | `1146.76ms` | Bad early loss and slow in the 100-step ablation. |

## Current Conclusions

1. `record` with the sparse attention gate disabled is the strongest local target-run result: it reaches val `< 3.28` at step `1404` in `638.155s`.
2. `output-metric-diag` does not currently justify its complexity: with the sparse attention gate disabled it reaches the same `3.2799` target 9 steps later and about `13.1s` slower than `record` no-gate.
3. Detached `record` projection does not look useful: it only reaches the target at step `1450`, with `665.314s` train time, slower than plain `record` no-gate.
4. `laplacian` is not competitive in this form. It runs at similar per-step speed to record/no-gate late in training, but ends far worse (`3.3001`) and never hits the `3.28` target.
5. `substoch-delta` is the clearest negative result: both the 100-step and 1385-step results have poor val loss. Unless there is a known bug in the implementation, this path should be deprioritized.
6. The sparse attention gate looks dispensable or harmful for the target metric in these runs. The no-gate `record` run is the best target-time result we have locally; the 200-step paired-head ablation also showed gate-off faster than gate-on.
7. Operationally, checkpointing failed for the laplacian and detached-record runs because the remote disk filled. Before any further checkpointed run, clear old checkpoints/caches or write checkpoints to a volume with at least `10G` free.
8. Paired-head record XSA is worth one longer no-gate follow-up only if we care about the better intermediate losses at steps `50-150`; at 200 steps it did not beat baseline no-gate final validation loss.

## Older XSA Performance Logs

| Log group | Configuration clarity | Result clarity | Result |
|---|---|---|---|
| `results/novita_1xh100_xsa_gs_2026-05-20/results/logs/novita-1xh100-baseline-smoke.txt` | Good enough: `xsa_mode=none`, one step | Good smoke result | step `1/1`, `53249ms`, peak/reserved `15093/15380 MiB` |
| `results/novita_1xh100_xsa_gs_2026-05-20/results/logs/novita-1xh100-value_no_gs.txt` | Mostly good; filename carries no-GS distinction | Good smoke result | step `1/1`, `42582ms`, peak/reserved `15581/15868 MiB` |
| `results/novita_1xh100_xsa_gs_2026-05-20/results/logs/novita-1xh100-value_gs.txt` | Mostly good; filename carries GS distinction | Good smoke result | step `1/1`, `46210ms`, peak/reserved `15391/15692 MiB` |
| `results/novita_1xh100_xsa_gs_2026-05-20/results/logs/novita-1xh100-model_no_gs.txt` | Mostly good; filename carries no-GS distinction | Good smoke result | step `1/1`, `59069ms`, peak/reserved `16824/17078 MiB` |
| `results/novita_1xh100_xsa_gs_2026-05-20/results/logs/novita-1xh100-model_gs.txt` | Mostly good; filename carries GS distinction | Good smoke result | step `1/1`, `88426ms`, peak/reserved `16835/16990 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/xsa_loss100_env210_latest/logs/novita-1xh100-env210-loss100-baseline.txt` | Good | Good | val `5.0314`, `230.87ms/step`, peak/reserved `37326/53002 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/xsa_loss100_env210_latest/logs/novita-1xh100-env210-loss100-value_no_gs.txt` | Mostly good; filename carries no-GS distinction | Good | val `5.0259`, `240.29ms/step`, peak/reserved `37393/54478 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/xsa_loss100_env210_latest/logs/novita-1xh100-env210-loss100-value_gs.txt` | Mostly good; filename carries GS distinction | Good | val `5.0410`, `241.34ms/step`, peak/reserved `37326/53922 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/value_xsa_perf100_env210_latest/logs/*.txt` | Good for mode; these were perf runs with no validation lines | Good timing/memory result | baseline `230.42ms`, value-GS `241.51ms`, value-no-GS `238.36ms` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/model_no_gs_loss100_env210_latest/logs/novita-1xh100-env210-loss100-model_no_gs.txt` | Good | Good | val `5.0355`, `280.47ms/step`, peak/reserved `41101/58166 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/model_lowrank_loss100_env210_latest/logs/novita-1xh100-env210-loss100-model_lowrank.txt` | Good | Good | val `5.0273`, `681.24ms/step`, peak/reserved `47563/64876 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/model_lowrank_head_loss100_env210_latest/logs/novita-1xh100-env210-loss100-model_lowrank_head.txt` | Good | Good | val `5.0479`, `707.34ms/step`, peak/reserved `56248/73358 MiB` |
| `results/novita_1xh100_xsa_loss100_2026-05-20/results/output_metric_loss100_env210_latest/logs/novita-1xh100-env210-loss100-output_metric.txt` | Good | Good | val `5.0575`, `306.54ms/step`, peak/reserved `39580/56682 MiB` |

## Logs With Missing Or Weak Result Records

| Log | Issue |
|---|---|
| `results/novita_1xh100_xsa_interp_2026-05-20/results/xsa_interp_latest/logs/novita-1xh100-xsa-interp-smoke.txt` | Has config and environment, but no obvious final metric, saved artifact path, or completion/result summary. |
| `results/novita_1xh100_xsa_interp_2026-05-20/results/xsa_interp_full_latest/logs/novita-1xh100-xsa-interp-4x65536.txt` | Same issue: no locally visible final metric, saved artifact path, or completion/result summary. |

## Recommendations

1. Treat canonical `.txt`/`canonical.log` files as the source of truth. Console logs should be considered secondary unless they include config lines.
2. For future runs, always print the exact command or normalized args near the top of the log. Current config lines are good, but command lines are usually absent.
3. For interpretation-only jobs, write a final summary line like `INTERP_RESULT path=... metrics=...` so they are auditable without loading generated arrays.
4. For suite console logs, add `RUN_START` / `RUN_DONE` around each mode. The runner scripts already do this, but several copied console logs lack the mode boundary lines needed to parse them standalone.
