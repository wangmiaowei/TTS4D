<div align="center">

# A Test-Time Solver for Uncertainty-Aware 4D Novel View Synthesis

[Miaowei Wang](https://wangmiaowei.github.io/)<sup>1</sup>,
[Zhi Cao](https://excitedbutter.github.io/)<sup>2</sup>,
[Jingwei Xu](https://github.com/xjwxjw)<sup>3</sup>,
[Oisin Mac Aodha](https://homepages.inf.ed.ac.uk/omacaod/)<sup>1</sup>,
[Amir Vaxman](https://avaxman.github.io/)<sup>1</sup>

<sup>1</sup>University of Edinburgh &nbsp; <sup>2</sup>Stanford University &nbsp; <sup>3</sup>NovaCore

[**Project page**](https://wangmiaowei.github.io/TTS4D/) &nbsp;|&nbsp;
[**Weights** 🤗](https://huggingface.co/miaoweiwang/TTS4D) &nbsp;|&nbsp;
[**Live demo**](https://wangmiaowei.github.io/TTS4D/demo.html) &nbsp;|&nbsp;
[**Slides**](https://huggingface.co/datasets/miaoweiwang/TTS4D-media/resolve/main/slides/TTS_talk.pptx)

<img src="docs/static/img/teaser.jpg" width="100%">

</div>

Transformer-based 4D novel view synthesis struggles to scale to many input views because softmax attention is quadratic in
the number of tokens. Test-time training (TTT) compresses the tokens into fast weights at linear cost, but writes them with
**one gradient step per chunk**: reads become blurred by interference and the memory depends on the token order.
Our **Test-Time Solver (TTS)** layer writes all input tokens into a single fast weight **in closed form**. It is
order-invariant, has a fixed size, trains about 2&times; faster than a comparable TTT memory, and, because the write is a
Bayesian linear regression, its posterior gives a per-pixel **epistemic uncertainty**; a small head adds the aleatoric part.

This repository contains the **inference code** and the released model weights (313 M parameters).

## Installation

```bash
git clone https://github.com/wangmiaowei/TTS4D.git
cd TTS4D
pip install -r requirements.txt      # PyTorch >= 2.4 (for nn.RMSNorm), einops, numpy, safetensors, huggingface_hub
```

## Quick start

```python
from tts4d.inference import from_pretrained, load_scene, inputs_of, encode, render_views

model = from_pretrained(device="cuda")                 # downloads tts4d.safetensors from the Hugging Face Hub
sc = load_scene("scene.npz")                           # posed input views (+ optional held-out targets), see below
memory = encode(model, inputs_of(sc, "cuda"))          # the test-time solve: one pass over the input views
out = render_views(model, memory, sc["tgt_c2w"], sc["tgt_K"], sc["tgt_time"], "cuda")
out["rgb"], out["std"], out["log_var_epi"], out["log_var_alea"]   # rendering, predicted std, the two uncertainty terms
```

The weights can also be downloaded once with `python scripts/download_weights.py` and loaded with
`load_model("weights/tts4d.safetensors")`.

**Scene format.** A scene `.npz` holds `inp_image` (uint8, V&times;3&times;H&times;W, H and W multiples of 8), `inp_c2w`
(OpenCV camera-to-world, V&times;4&times;4), `inp_K` (fx, fy, cx, cy in pixels) and `inp_time` (seconds), and optionally the
same `tgt_*` arrays for held-out targets. `tts4d/geometry.py` (`normalize_scene`) brings cameras into the model frame
(first input camera = identity, scale from the inputs only).

The model runs in bfloat16 autocast with the solve in float32; 32 views at 360&times;480 need about 9 GB of GPU memory.

## Notebooks

| notebook | what it shows |
|---|---|
| [`notebooks/fit_one_scene.ipynb`](notebooks/fit_one_scene.ipynb) | one dynamic scene (DyCheck *handwavy*, 32 views): solve once, render held-out views, uncertainty vs. error, order invariance, a novel camera path (executed, outputs included) |
| [`notebooks/tts_vs_ttt_memory.ipynb`](notebooks/tts_vs_ttt_memory.ipynb) | **TTS vs. TTT** on the write operation alone, no weights or data needed: recall of written tokens as the number of views grows, sensitivity to the token order, where each read comes from, and the speed of a training step |

## How the model works, in code

```
tts4d/
  tts_layer.py   TTS layer: closed-form centred & scaled ridge solve, read, epistemic scalar u(q)
  model.py       TTS4D: patch tokens (Plücker rays + time + RGB), 24 layers, RGB decoder, uncertainty head
  blocks.py      one layer = within-view self-attention + TTS layer (all views) + MLP
  rope.py        2-D + time rotary embedding of the TTS layer's q / k
  geometry.py    Plücker rays, scene normalisation (first input camera = identity, input-only scale)
  inference.py   model / scene loading, batched rendering, PSNR, training-free grid removal, uncertainty colouring
configs/tts4d.json   hyper-parameters of the released model
```

* `encode(inputs)` runs the input views through the network. In each of the 24 TTS layers the input tokens are written into
  **one fast weight** by a closed-form ridge regression (`TTSLayer.solve`): five sufficient statistics are summed over
  tokens, centred, scaled and solved with one Cholesky factorization. The state has a fixed size for any number of input
  views and does not depend on their order.
* `render(memory, queries)` renders any number of query views (camera + time) from that state; query tokens only **read**
  the memory, so a scene is solved once and then rendered from any camera and time.
* Uncertainty: every TTS layer also returns the posterior variance `u_l(q)` of the query read. The epistemic log-variance is
  `U_epi = beta_0 + sum_l beta_l log(u_l / u_bar_l)`; the total adds a learned aleatoric term, and `exp(U/2)` is the
  predicted standard deviation.

Training code is not included.

## Citation

```bibtex
@article{wang2026tts,
  title   = {A Test-Time Solver for Uncertainty-Aware 4D Novel View Synthesis},
  author  = {Wang, Miaowei and Cao, Zhi and Xu, Jingwei and Mac Aodha, Oisin and Vaxman, Amir},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements and license

The network layout (patch tokens with Plücker rays, within-view attention interleaved with a fast-weight layer across views)
follows the publicly released [LaCT](https://github.com/a1600012888/LaCT) and FSM code bases; the TTS layer, the uncertainty
head and this implementation are ours. The code is released under the [MIT License](LICENSE).
