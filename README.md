# Logit Classifier

A local zero-shot classifier. One state and a set of questions go in, and a calibrated
probability per option comes out. No text is generated, since every answer is read from
the logits at a single token position.

The request format is TypeSafe's System One, so a Jev request body works unchanged. The
default model reads images as well as text.

## Question Types

| Type | You declare | You get back |
|---|---|---|
| `choice` | a map of option name to description | the winning name, a probability per option, a confidence, and an abstain |
| `score` | an ordered list of level descriptions | a weighted level, a probability per level, the legend, and a confidence |
| `noul` | one statement, with optional true and false descriptions | one probability that the statement is true |

A `choice` holds up to 255 options. A `score` holds 2 to 10 levels. One request holds up
to 256 questions, and every question is answered against the same state in one pass.

A branch is one prompt whose last token position carries a distribution. A question needs
one branch, or several when it holds more than 52 options or scores each level alone.

`abstain` is how strongly the model wants none of your options. Every choice question is
offered an extra label named `none of these`, and that label's mass comes back as
`abstain` instead of in `probabilities`. So `probabilities` covers your own options and
sums to 1.

`confidence` measures how peaked the answer is, using TypeSafe's published formulas.

Ask one `noul` per fact when several answers can be true at once. A `choice` normalizes
over its options, so one winner takes nearly all the mass.

## Install

The package needs Python 3.12 or newer.

```
pip install "logit-classifier[hf,service]"
```

| Extra | Adds | For |
|---|---|---|
| none | numpy | passing a model you already loaded |
| `hf` | torch, torchvision, transformers, accelerate, safetensors, huggingface-hub, pillow | loading the weights in this process |
| `service` | fastapi, uvicorn, pillow | the HTTP endpoint and the browser page |

The `hf` extra needs an NVIDIA GPU with at least 10 GB of free VRAM. The weights are
about 8 GB per model and download on first use.

They go to the Hugging Face cache by default. Point `models_dir` at a folder to keep them
beside your project instead, which is what the examples do.

```python
Config(models_dir=Path("models"))
```

`LOGIT_MODELS_DIR` sets the same folder for every process, and `HF_HOME` moves the
Hugging Face cache itself.

## Models

| | Qwen/Qwen3-VL-4B-Instruct | Qwen/Qwen3-4B-Instruct-2507 |
|---|---|---|
| reads images | yes | no |
| better at | choice and score | noul |

`Qwen3-VL-4B-Instruct` is the default. Set `LOGIT_MODEL_ID` to use the other one. Any Qwen
chat model loads, and a model with no fitted temperature gets 2.5.

## Python API

```python
from logit_classifier import Classifier, Config, load_model, parse_request

config = Config()
backend = load_model("Qwen/Qwen3-VL-4B-Instruct", config)
classifier = Classifier(config, backend=backend)

request = parse_request({
    "state": "the payment failed again",
    "questions": {
        "urgent": {"type": "noul", "instructions": "The message is urgent"},
    },
})
response, diagnostics = classifier.classify(request)
print(response.to_dict())
```

`load_model` loads the weights through the `hf` extra and returns a `Backend`.
`Classifier(Config())` calls it for you when you pass no backend.

Loading each model yourself is what lets one script compare several. The fitted
temperature follows the backend, so every model is scored with its own value.

```python
for model_id in ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"):
    classifier = Classifier(config, backend=load_model(model_id, config))
```

## HTTP API

```
logit-classifier serve
```

Run it as `python -m logit_classifier serve` when the scripts directory is not on PATH.

`POST /v1/systemone` takes one `state` and a map of questions.

```json
{
  "state": "I have been trying to connect my Stripe account for 3 days and it keeps failing.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": null
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated the customer appears",
      "criteria": ["Calm", "Frustrated but civil", "Very angry"]
    },
    "is_urgent": { "type": "noul", "instructions": "The message conveys urgency" }
  }
}
```

Each answer comes back under the key you gave its question. The keys are never sent to
the model.

```json
{
  "model": "logit-classifier-0.2.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.9518,
      "probabilities": { "billing": 0.0237, "technical": 0.9679, "sales": 0.0084 },
      "abstain": 0.0068
    },
    "frustration": {
      "type": "score",
      "score": 1.0199,
      "confidence": 0.9583,
      "legend": { "0": "Calm", "1": "Frustrated but civil", "2": "Very angry" },
      "probabilities": { "0": 0.004, "1": 0.9722, "2": 0.0238 }
    },
    "is_urgent": { "type": "noul", "noul": 0.9924 }
  },
  "usage": { "input_tokens": 210, "output_tokens": 3 }
}
```

Add `?diagnostics=1` to also return the label mass, the branch count, the latency and the
calibration settings.

`GET /v1/models` lists the served model. `GET /healthz` reports whether the weights are
loaded. Open `http://127.0.0.1:8077/` for a browser form that builds a request and draws
each probability as a bar.

## Images

A `state` that is a mapping may carry one image under `image` or `screenshot`, as a file
path, a data URL or bare base64. Everything else in that mapping is rendered as text
beside the picture. The HTTP service accepts a data URL or base64, and a file path works
only in process.

```json
{ "state": { "image": "tile-3.png", "caption": "tile 3 of 6" },
  "questions": { "sky": { "type": "noul", "instructions": "This crop shows the night sky" } } }
```

The image is encoded once for the entire request. So asking several questions about one
picture costs little more than asking one.

A model with no vision tower rejects the request rather than ignoring the picture.

## Examples

Each script runs on its own.

| Script | Shows |
|---|---|
| `examples/quickstart.py` | one request with all three question types |
| `examples/question_types.py` | every field each answer type returns |
| `examples/images.py` | a picture, and a picture with text beside it |
| `examples/compare_models.py` | two models answering the same questions in one command |
| `examples/http_client.py` | calling the service over HTTP |
| `examples/own_backend.py` | implementing the backend port, with no weights or GPU |

## Configuration

Every setting reads from the environment at startup. `logit-classifier config` prints what
they produce.

| Variable | Default | Effect |
|---|---|---|
| `LOGIT_MODEL_ID` | `Qwen/Qwen3-VL-4B-Instruct` | any Qwen chat model, with or without vision |
| `LOGIT_MODELS_DIR` | the Hugging Face cache | where downloaded weights are kept |
| `LOGIT_DEVICE` | `cuda` | torch device |
| `LOGIT_TEMPERATURE` | the model's fitted value | divides the logits before the softmax |
| `LOGIT_PRIOR_DEBIAS` | `1` | set to `0` to skip the label prior |
| `LOGIT_BATCH_BRANCHES` | `1` | set to `0` for one forward pass per branch |
| `LOGIT_SCORE_METHOD` | `joint` | set to `independent` to judge each level alone |
| `LOGIT_ABSTAIN` | `1` | set to `0` to drop the `none of these` label below 52 options |
| `LOGIT_CALIBRATION_PATH` | `calibration.json` | where the service stores the running prior |

## Calibration

Two corrections are applied before the softmax.

The label prior is the model's standing preference for the token `A` over `B`. It is
learned as a running mean over served requests, so it needs no labelled data or extra
forward passes. It stays unused until a bucket has 32 observations, so the first requests
after a cold start can answer a little differently from later ones.

The temperature is fitted per model and divides the logits. Both shipped values were
fitted on Banking77 rather than on your data, so refit yours when the absolute numbers
matter.

The library holds the prior in memory and writes no file. The service stores it in
`calibration.json`. Pass `Config(calibration_path=...)` to keep it across runs from the
library. A stored prior carries a fingerprint of the model and the prompt shape, and
changing either one starts the prior over.

## Determinism

Nothing is sampled, so the same request returns bitwise identical logits.

Batch composition and padding length both change the low bits of a bfloat16 forward pass,
and both are pure functions of the request. So the same question asked inside two
different requests can differ slightly. Set `LOGIT_BATCH_BRANCHES=0` to score each branch
alone, which makes a question independent of the questions sent with it and costs one
forward pass per branch. `ComfyClipBackend` reads no environment variable, so it takes
`batch_branches=False` as a keyword instead.

A forward pass needs several process-global torch settings held at known values. Torch
exposes none of them as a call argument, so the backend sets them around each pass and
puts the host's values back after. This way a host that shares the interpreter keeps its
own settings everywhere outside a pass.

## Options Above 52

Options are labeled `A` to `Z` then `a` to `z`, which gives 52 single token labels. A
question with more options is split into groups, and each group carries an escape label so
the groups can be weighed against each other. Accuracy is lower on that path and `abstain`
is much weaker, so keep a question under 52 options where you can.

## Speed

On a warm RTX 3090 Ti a short record takes about 140 ms whatever it asks, up to about 8
questions, because two forward passes dominate. So send every question for a record in one
request. Keep a record under about 8,000 tokens, since a longer state falls off the
prefill ceiling.

## ComfyUI Node Packs

A node pack installs the bare package without extras, so pip is never asked to resolve
torch beside ComfyUI's own CUDA build. The pack then passes its own backend over a model
ComfyUI already loaded, and this library never imports torch or transformers.

A node pack uses its own backend in place of `load_model`. Everything after that is the
same call.

```python
from logit_classifier import Backend, Classifier, Config, verify_backend

assert isinstance(my_backend, Backend)
label_ids = verify_backend(my_backend)
classifier = Classifier(Config(), backend=my_backend)
```

`Backend` declares `model_id`, `label_ids` and `sees_images`, plus `render`, `encode`,
`encode_prefix` and `score`. `model_id` is the string the fitted temperature is looked up
by, so a backend over the text model must report the text model. `Backend` is runtime
checkable, so a host can check its own object before passing it. `verify_backend` proves
the render and label contracts and returns one token id per label.

`COMFY_SOCKET_TYPE` holds the ComfyUI socket name for a loaded classifier.
`examples/own_backend.py` implements the entire port.
`logit_classifier.backends.hf.HFBackend` is the transformers implementation to read
against.

`ComfyClipBackend` is the backend over a Qwen3-VL text encoder that the workflow already
loaded, such as the one Krea 2 uses. So no second model is loaded into VRAM.

```python
from logit_classifier import Classifier, Config, NoulQuestion, SystemOneRequest
from logit_classifier.backends.comfy_clip import ComfyClipBackend

classifier = Classifier(Config(), backend=ComfyClipBackend(clip))
question = NoulQuestion(instructions="This image visibly contains a dragon")
request = SystemOneRequest(state="", questions={"dragon": question})
response, _ = classifier.classify(request, image=image)
```

The `image` keyword takes an image the host already decoded, such as a ComfyUI IMAGE of
shape `[1, H, W, 3]`. An empty state asks about the image alone. Questions share one forward
pass of up to 4,096 tokens, and the image counts toward that limit. Any text encoder other
than Qwen3-VL raises `UnsupportedModelError`.

## Differences From Jev

This project matches the System One request and response format and the documented limits.
It is not affiliated with TypeSafe. Its numbers do not match Jev's.

Jev is trained for calibrated probabilities. This project reads them from a general model,
so the choices agree more often than the confidences do.

Jev judges a score level without its number or its neighbours. This project judges all
levels together by default. Set `LOGIT_SCORE_METHOD=independent` for the documented
behavior.

Jev publishes status codes but no error body. The error shape here is our own.

## Development

```
uv sync --all-extras
uv run pytest tests/test_unit.py
uv run pytest tests/test_model.py -m model
uv run ruff check .
uv run mypy
```

The unit tests need no weights. The model tests load the GPU and take about 15 seconds.

`FINDINGS.md` is the measurement record. Every number there names the script in
`tests-AB/` that produced it.

## License

GPL-3.0. See `LICENSE`.
