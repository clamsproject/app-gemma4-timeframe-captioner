# Gemma 4 Captioner

A CLAMS app that captions video TimeFrames with a
[Gemma 4](https://huggingface.co/google/gemma-4-E4B-it) vision-language model,
run with [llama.cpp](https://github.com/ggml-org/llama.cpp) from a GGUF file.
It uses the SDK's experimental `ClamsGGUFPromptableApp` backend: the app starts
its own `llama-server` process on localhost with the pinned model and stops it
when it exits.

The app mirrors
[`app-qwen35-timeframe-captioner`](https://github.com/clamsproject/app-qwen35-timeframe-captioner)
(same default prompts and output), so the two can run on the same MMIF to
compare the torch and llama.cpp backends. It has no `parallelPrompts`
parameter: llama-server's slot count sets how many prompts run together (see
below).

The `model` runtime parameter selects the Gemma 4 size and quantization from
the official [ggml-org](https://huggingface.co/ggml-org) GGUF conversions of
the instruction-tuned models, from E2B to 31B, at BF16, Q8_0 or Q4_0; the
default is `ggml-org/gemma-4-E4B-it-GGUF:Q8_0`.
The context size is not fixed: llama-server sizes it to the free GPU memory,
so how many frames fit in one prompt depends on the GPU (the app logs the
size at startup).
See the GPU memory table in [`metadata.py`](metadata.py) before selecting one.
Gemma 4 can reason ("think") before answering.
By default the app runs it without reasoning and strips any reasoning block
from the caption.
Set `useReasoning` to enable reasoning and keep the trace (stored on the
caption's `modelReasoningTrace` property); the trace comes out of the same
`maxNewTokens` budget as the caption, so raise `maxNewTokens` when enabling it.

The canonical user-facing documentation (app description, runtime parameters,
I/O specs, model list, GPU memory estimates) lives in
[`metadata.py`](metadata.py), rendered at the
[CLAMS App Directory](https://apps.clams.ai).
For general CLAMS app usage, see the
[CLAMS App Manual](https://apps.clams.ai/clamsapp).

## System requirements

- The `llama-server` executable from llama.cpp, recent enough for Gemma 4.
  The container image includes a CUDA build for NVIDIA GPUs (tested on
  Maxwell, Pascal and Ampere). Outside the container, install a llama.cpp
  build for your hardware (e.g. Metal on a Mac) and put `llama-server` 
  on the `PATH` or set `LLAMA_SERVER` to its path. Without a GPU, it runs 
  on CPU, much slower.
- GPU memory for the selected `model`; see [`metadata.py`](metadata.py)
  (`est_gpu_mem_min` is the smallest variant, `est_gpu_mem_typ` the
  default). llama-server splits a model larger than one GPU over the
  visible GPUs.
- `ffmpeg` (used by `mmif-python[cv]` for video frame extraction).
- In the container, mount a volume at `/cache`. It holds the downloaded model
  and the CUDA kernels the driver compiles at the first start on GPUs the
  image has no prebuilt kernels for (about a minute on Maxwell and Pascal);
  without the volume, every new container compiles them again.

## llama-server settings

The app sets two llama-server settings (`LLAMA_ARGS` in [`app.py`](app.py)):
a micro-batch of 2048 tokens (`LLAMA_ARG_UBATCH`), so an image gets its full
token budget, and one slot (`LLAMA_ARG_N_PARALLEL`), so a caption does not
depend on the other prompts decoded with it. Everything else (GPU offload,
context size, threads) comes from llama-server's defaults, which place what
fits on the GPUs and keep the rest in system memory. To change any setting,
set llama.cpp's own `LLAMA_ARG_*` environment variable for the app process
(for example `LLAMA_ARG_CTX_SIZE=16384`); the environment takes precedence
over the app's values. The settings in effect are logged at startup, but not
recorded in the output MMIF.
