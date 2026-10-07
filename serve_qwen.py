# ---
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with Qwen3 and vLLM

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# This example is intended to demonstrate the basics of deploying LLM inference on Modal.
# For more on how to optimize performance, see
# [this guide](https://modal.com/docs/guide/high-performance-llm-inference)
# and check out our
# [LLM Engineer's Almanac](https://modal.com/llm-almanac).

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

import json
import os
from typing import Any

import aiohttp
import modal

# Which GPU to serve on, picked at deploy time: GPU=L40S modal deploy serve_qwen.py
# Each GPU gets its own app (and URL) so they can be benchmarked side by side.
# The value is baked into the image so the container sees the same setting.
GPU = os.environ.get("GPU", "A10G")

# Which weights to serve, also picked at deploy time: VARIANT=awq KV_DTYPE=fp8 modal deploy ...
#   bf16: full 16-bit weights (~16GB)
#   fp8:  8-bit weights (~8GB); fast on GPUs with FP8 hardware (L40S, H100), not A10G
#   awq:  4-bit weights (~5GB); runs on any recent NVIDIA GPU
# KV_DTYPE=fp8 also stores the KV cache (each conversation's working memory) in 8 bits,
# so about twice as many conversations fit.
VARIANT = os.environ.get("VARIANT", "bf16")
KV_DTYPE = os.environ.get("KV_DTYPE", "auto")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            "VLLM_LOG_STATS_INTERVAL": "1",  # more frequent metrics logging
            "GPU": GPU,
            "VARIANT": VARIANT,
            "KV_DTYPE": KV_DTYPE,
        }
    )
)

# ## Download the model weights

# We'll serve [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), an 8 billion parameter open model
# from Alibaba, as the persona (caller) model for Mini Bluejay.
# In bf16 its weights take ~16GB, so a single 24GB A10G holds the model plus a KV cache
# (the memory where in-flight requests keep their context) big enough to batch a few
# dozen short conversations at once.


# Qwen's official checkpoints, pinned so a repo update can't change results under us.
MODELS = {
    "bf16": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "fp8": ("Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3"),
    "awq": ("Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79"),
}
MODEL_NAME, MODEL_REVISION = MODELS[VARIANT]
MAX_MODEL_LEN = 8192  # our calls are ~1-2k tokens; a shorter max leaves more GPU memory for batching

# Although vLLM will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk.
# For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Configuring vLLM

# ### Trading off fast boots and token generation performance

# vLLM has embraced dynamic and just-in-time compilation to eke out additional performance without having to write too many custom kernels,
# e.g. via the Torch compiler and CUDA graph capture.
# These compilation features incur latency in exchange for lowered latency and higher throughput during generation.
# This latency is typically tens of seconds to a few minutes, reduced to about ten seconds when loaded from the cache.
# We make this trade-off controllable with the `FAST_BOOT` variable below.

FAST_BOOT = False

# If you're running an LLM service that frequently scales from 0 (frequent ["cold starts"](https://modal.com/docs/guide/cold-start))
# you might want to set this to `True`, or consider [GPU memory snapshots](https://modal.com/docs/guide/memory-snapshots).
# It's also useful to set this when you're iterating on the server configuration.

# If you're running an LLM service that usually has multiple replicas running, then set this to `False` for improved performance.

# See the code below for details on the parameters that `FAST_BOOT` controls.

# ### Model-specific configuration

# Qwen3 needs one extra flag: a reasoning parser, so any "thinking" text is kept out of
# the reply. We turn thinking off per request anyway, since a caller should just talk.


# For more on the performance you can expect when serving your own LLMs, see
# [our LLM engine performance benchmarks](https://modal.com/llm-almanac).

# ## Build a vLLM engine and serve it

# The class below spawns a vLLM instance listening at port 8000, serving requests to our model.
# We wrap it in the [`@app.server`](https://modal.com/docs/guide/servers) decorator
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the process is ready to listen on the configured port.


APP_NAME = f"mini-bluejay-vllm-{GPU.lower()}"
if VARIANT != "bf16":
    APP_NAME += f"-{VARIANT}"
if KV_DTYPE != "auto":
    APP_NAME += f"-kv{KV_DTYPE}"
app = modal.App(APP_NAME)

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000
ROUTING_REGION = "us-east"


@app.server(
    image=vllm_image,
    gpu=f"{GPU}:{N_GPU}",  # A10G 24GB / L40S 48GB: weights take ~16GB, the rest is KV cache
    scaledown_window=5 * MINUTES,  # idle time before the GPU shuts down (and stops billing)
    startup_timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    routing_region=ROUTING_REGION,
    # Let vLLM do the queueing: with one container, Modal should pass every request through.
    target_concurrency=256,
    max_containers=1,  # one GPU only, so the benchmark measures a single replica
    unauthenticated=True,  # to make the endpoint publicly accessible
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--uvicorn-log-level=info",
            "--async-scheduling",
        ]

        # enforce-eager disables both Torch compilation and CUDA graph capture
        # default is no-enforce-eager. see the --compilation-config flag for tighter control
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        # Qwen3 can "think" before answering; this parser keeps that text out of the reply.
        # Clients turn thinking off per request with chat_template_kwargs={"enable_thinking": False}.
        cmd += ["--reasoning-parser", "qwen3"]

        # quantized weights are detected from the checkpoint; the KV cache is opt-in
        if KV_DTYPE != "auto":
            cmd += ["--kv-cache-dtype", KV_DTYPE]

        print(*cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy vllm_inference.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--mini-bluejay-vllm-server.us-east.modal.direct`.

# To interact with the API programmatically in Python, we recommend the `openai` library.

# See the `client.py` script in the examples repository
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible)
# to take it for a spin:

# ```bash
# # pip install openai==1.76.0
# python openai_compatible/client.py
# ```


# ## Testing the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server. As opposed to Modal Functions, however
# when a Server has no active containers, requests will be rejected with a 503 Service Unavailable status.
# Therefore, we have to handle this manually in the client code.

# If you execute the command

# ```bash
# modal run vllm_inference.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=15 * MINUTES, content=None, twice=True):
    import asyncio
    import time

    url = await Server.get_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    # python.org Python on macOS ships without system root certs; use certifi's bundle
    import ssl

    import certifi

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(base_url=url, connector=connector) as session:
        print(f"Running health check for server at {url}")
        deadline = time.time() + test_timeout - 1 * MINUTES
        while time.time() < deadline:
            async with session.get(
                "/health", timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    break
                if resp.status == 503:
                    await asyncio.sleep(1)
                    continue
                assert False, (
                    f"Failed health check for server at {url}: HTTP {resp.status}"
                )
        else:
            assert False, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    # personas should answer directly, not think first (faster and cheaper)
    payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            delta = chunk["choices"][0]["delta"]
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if content:
                print(content, end="")
            else:
                print("\n", chunk)
    print()


# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible):

# ```bash
# modal run openai_compatible/load_test.py
# ```
