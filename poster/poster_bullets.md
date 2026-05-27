# INTRODUCTION
* Exclusive Self Attention (XSA) (\cite{zhai2026exclusive}) is a modification to the self-attention operation in transformer-based LLMs.

* For each attention head, it removes the component along the current token's value vector from the attention output. For head \(h\) at position \(i\), standard attention computes
\[
y_{h,i}=\sum_{j\leq i} a_{h,ij}v_{h,j},
\]
whereas XSA instead computes
\[
z_{h,i}
=
\left(
I-\frac{v_{h,i}v_{h,i}^{\top}}{\|v_{h,i}\|^2}
\right)y_{h,i}.
\]

* The intuition behind this is that attention should model contextual information, while information about the current token is already available through the residual stream and subsequent feedforward network (FFN).

* XSA also functions as an _implicit attention sink_ since it removes all attention component along $v_i$, hence models can learn to discard attention mass by attending to the self-value. This allows tokens to allow to "nothing" without adding an explicit sink token.

# DEMONSTRATING EXISTENCE OF THE PHENOMENON

* In an 11-layer GPT-2-style model trained on FineWeb we observe $v_i$-alignment across heads and layer depths

[PLOTS OF THIS]

# TRAINING AN XSA MODEL

* We devised various ways of implementing XSA:

EQNs for: value space, model space, output metric diag

* value space was the XSA default, output metric diag is promising, more principled, and performs similarly, but is hard to implement with faster runtime

[VALUE SPACE XSA WORKS, TABLE SHOWING STEPS/RUNTIME IMPROVEMENT]

# INTERP

* We wanted to better understand the mechanics behind why XSA works

Attention table
| - | a_ii (~sink)| y_ctx self-aligned proportion |
| noXSA | low | low |
| XSA   | high | high |

=> interpretation: the model knows that a_ii / v_i-aligned will be discarded, and allocates attention mass towards these (sinking hypothesis shown!)

* Causal interventions: we took a model trained to completion with XSA and perturbed attention scores at eval time to tease out the mechanisms behind XSA

| Intervention | Overall \(\Delta \ell\) | Interpretation |
|---|---:|---|
| `no_xsa_all` | \(-1.6383\) | XSA is doing a very large amount of causal work. |
| `shuffle_proj_g1_Lall` | \(-0.9055\) | Removing arbitrary/shuffled value projection is also bad. |
| `force_diag_c0_Lall` | \(-0.3693\) | Removing diagonal attention mass globally hurts. |
| `context_addback_e1_Lall` | \(-0.9854\) | Adding back context-derived \(v_i\)-parallel projection is very damaging. |
| `diag_addback_e0p5_Lall` | \(-0.1069\) | Adding back half the diagonal value contribution after XSA only mildly hurts. |

* Loss bucket tables?

# IMPROVING XSA IN NANOGPT SPEEDRUN - there's something else afoot...

* We discovered that NANOGPT speedrun had a quasi-attention sink function performed by "Sparse attention gates". We showed that XSA functions well as an attention sink, so we suspected that this might now be redundant

[PLOT SHOWING WE ARE THE GOATS]

# Conclusion

* confirms the intuition of attention as a contextual mixing operator
* confirms that XSA provides an efficient implicit attention sink
* cannot be applied blindly, not every layer needs full-strength XSA
