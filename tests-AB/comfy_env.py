"""ComfyUI bootstrap for a standalone A/B harness that runs core's code with no ComfyUI server.

The Logit Tagger's and ContextAnchoredTileRefine's `tests-AB/ab_env.py` call this module with their own
machine paths. It must never run inside a node, since it rewrites `sys.argv`, `sys.path`, `os.environ` and
comfy globals that the running server owns.

`bootstrap()` runs before the first comfy or folder_paths import, since those names resolve to whichever
source tree sits on sys.path and folder_paths fixes its base path at import.
"""

import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# main.py runs these stages once per process, so a second bootstrap() changes nothing.
_bootstrapped = False


def _enable_dynamic_vram() -> tuple[bool, str]:
    """Bring up core's DynamicVRAM (comfy_aimdo) as main.py does, and return (enabled, why).

    Core takes its CUDA graph decode and its --fast-disk staging only with DynamicVRAM on, so without it the
    harness runs a slower memory path than the app. Mirrors main.py's control.init and init_devices blocks.
    """
    import comfy_aimdo.control
    from comfy.cli_args import args, enables_dynamic_vram

    simple_vram_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024**3)
    device_headroom = int(args.vram_headroom * 1024**3)
    supported = False
    initialized = False

    if not enables_dynamic_vram():
        return False, "the parsed ComfyUI flags turn DynamicVRAM off"
    # main.py calls control.init before anything initializes a torch device.
    try:
        comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom, nvml_pressure=not args.disable_nvml_pressure)
    except TypeError:
        # comfy-aimdo 0.4.10 protocol.
        try:
            comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom)
        except TypeError:
            # comfy-aimdo 0.4.9 protocol.
            comfy_aimdo.control.init()

    import comfy.memory_management
    import comfy.model_management as model_management
    import comfy.model_patcher

    supported = model_management.is_nvidia() or (model_management.is_amd() and model_management.rocm_version >= (7, 14))
    if not (args.enable_dynamic_vram or supported):
        return False, "the GPU is neither NVIDIA nor AMD with ROCm 7.14 or later"
    if not args.enable_dynamic_vram and model_management.torch_version_numeric < (2, 8):
        return False, f"torch {model_management.torch_version_numeric} is older than 2.8"
    try:
        initialized = comfy_aimdo.control.init_devices(
            (device.index, device_headroom) for device in model_management.get_all_torch_devices()
        )
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        initialized = comfy_aimdo.control.init_devices(device.index for device in model_management.get_all_torch_devices())
    if not initialized:
        return False, "comfy_aimdo reported no working install"
    # main.py's branch for the default INFO console log level.
    comfy_aimdo.control.set_log_info()
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    return True, "comfy_aimdo initialized every torch device"


def bootstrap(comfy_root: Path, base_directory: Path, extra_args: Sequence[str] = ()) -> None:
    """Import comfy from comfy_root, parse core's flags against base_directory and enable DynamicVRAM."""
    global _bootstrapped
    root = comfy_root.resolve()
    base = base_directory.resolve()
    spec = None
    origin = None
    saved_argv = sys.argv
    dynamic_vram = False
    why = ""

    if _bootstrapped:
        return
    if not (root / "comfy").is_dir():
        raise SystemExit(f"ComfyUI root {root} has no 'comfy' directory. Pass the server source, not the base path.")
    if not base.is_dir():
        raise SystemExit(f"ComfyUI base directory {base} does not exist. Pass the folder that holds models.")
    # cli_args parses argv exactly once, at its own import, and says nothing when it already ran.
    if "comfy.cli_args" in sys.modules:
        raise SystemExit("comfy.cli_args was imported before bootstrap(). Nothing may import comfy or folder_paths first.")

    # main.py sets this before native allocators load.
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    if sys.path[:1] != [str(root)]:
        sys.path.insert(0, str(root))

    # An installed 'comfy' package, such as the one comfy-cli ships, beats the source tree whatever the
    # sys.path order.
    spec = importlib.util.find_spec("comfy.sd")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or origin.parent != (root / "comfy").resolve():
        raise SystemExit(f"A different 'comfy' package shadows the ComfyUI source at {root}: comfy.sd is {origin}")

    # cli_args parses argv only after enable_args_parsing, as main.py does. A synthetic argv keeps the
    # harness's own flags away from ComfyUI's parser.
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [saved_argv[0], "--base-directory", str(base), *extra_args]
    try:
        import folder_paths  # noqa: F401
    finally:
        sys.argv = saved_argv

    # main.py imports it before torch loads. It picks the app's cudaMallocAsync allocator from the parsed flags.
    import cuda_malloc  # noqa: F401

    # A missing comfy_aimdo or a raising init leaves the legacy ModelPatcher, and the warning below says so.
    try:
        dynamic_vram, why = _enable_dynamic_vram()
    except Exception as error:
        dynamic_vram, why = False, f"{type(error).__name__}: {error}"
    _bootstrapped = True
    print(f"DynamicVRAM {'on' if dynamic_vram else 'OFF, so the harness runs a different memory path than the app'}: {why}",
          flush=True)


def end_node_execution() -> None:
    """Run the cleanup core's executor runs after every node under DynamicVRAM, in execution.py after get_output_data.

    A harness that runs several node executions in one process calls it between them, as the app would.
    """
    import comfy.memory_management

    if not comfy.memory_management.aimdo_enabled:
        return
    import comfy.model_management
    import comfy.model_prefetch
    import comfy_aimdo.model_vbar

    comfy.model_prefetch.cleanup_prefetch_queues()
    comfy.model_management.reset_cast_buffers()
    comfy_aimdo.model_vbar.vbars_reset_watermark_limits()


def load_clip(name: str, clip_type: str = "krea2") -> Any:
    """Load a text encoder from models/clip as core's CLIPLoader does with device default."""
    import comfy.sd
    import folder_paths

    clip_path = ""
    clip_enum = comfy.sd.CLIPType.__members__.get(clip_type.upper())

    if clip_enum is None:
        raise ValueError(f"Unknown CLIP type {clip_type!r}. comfy.sd.CLIPType holds {sorted(comfy.sd.CLIPType.__members__)}.")
    clip_path = folder_paths.get_full_path_or_raise("text_encoders", name)
    return comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_enum,
        model_options={},
    )
