# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Unified CARLA + Alpamayo launcher: pick an Alpamayo version and a loop mode.

"""CarlaMayo unified entrypoint.

Select the Alpamayo release with ``--version {1,1.5,2}`` and the CARLA loop with
``--loop {open,closed,live-open}``:

* ``open``       Replay a recorded dataset through the model (offline, no driving).
* ``closed``     The model drives the ego vehicle via a PID follower.
* ``live-open``  CARLA autopilot drives; the model runs live and is observed open-loop.

The thin ``carlamayo_open_loop.py`` / ``carlamayo_closed_loop.py`` scripts call this
module with a preset loop for backward compatibility.
"""

import argparse

from module.adapters import SUPPORTED_VERSIONS, get_adapter

LOOPS = ("open", "closed", "live-open")


def build_parser(preset_loop=None):
    parser = argparse.ArgumentParser(
        description="Run CARLA + Alpamayo in open / closed / live-open loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        dest="version",
        required=True,
        choices=SUPPORTED_VERSIONS,
        help="Alpamayo model version to run (required).",
    )
    if preset_loop is None:
        parser.add_argument(
            "--loop",
            required=True,
            choices=LOOPS,
            help="CARLA loop mode.",
        )

    # Shared model-loading options.
    parser.add_argument(
        "--quantization", dest="quantization", action="store_true",
        help="Use 4-bit quantized model instead of full precision.",
    )
    parser.add_argument(
        "--oom-free", dest="oom_free", action="store_true",
        help="Use OOM-free CPU<->GPU demand layering (Alpamayo 1.5 only).",
    )
    parser.add_argument("--oom-free-headroom-gb", type=float, default=None,
                        help="OOM-free: VRAM (GB) reserved for activation spikes.")
    parser.add_argument("--oom-free-margin", type=int, default=None,
                        help="OOM-free: safety margin subtracted from resident VLM layers.")
    parser.add_argument("--oom-free-resident", type=int, default=None,
                        help="OOM-free: force this many GPU-resident VLM layers.")
    parser.add_argument("--device-map", default="auto",
                        help='Model device_map passed to from_pretrained.')
    parser.add_argument("--max-gpu-memory-gb", type=float, default=None,
                        help="Cap model memory per GPU; layers that do not fit are offloaded "
                             "to CPU RAM (slow). Use for Alpamayo 2 on a <80 GB GPU.")
    parser.add_argument("--cuda-linalg-library", choices=("default", "cusolver", "magma"),
                        default="magma", help="Preferred CUDA linalg backend.")
    parser.add_argument("--output-video", default=None,
                        help="Output video path. Defaults to a per-loop name.")

    # CARLA connection (override to run alongside other simulators on one host).
    parser.add_argument("--carla-host", default="localhost", help="CARLA server host.")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA RPC port.")
    parser.add_argument("--tm-port", type=int, default=8000,
                        help="CARLA Traffic Manager port.")

    # Open-loop options.
    parser.add_argument("--data-root", default="carla_data",
                        help="[open] Dataset root with trajectory.json and camera folders.")

    # Closed-loop / live-open options.
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="[closed, live-open] Non-blocking background inference worker.")
    parser.add_argument("--pygame-ui", action="store_true",
                        help="[closed] Pygame camera UI with prompt input and pause/resume.")
    parser.add_argument("--mode", choices=("normal", "navigation", "vqa"), default="normal",
                        help="[closed] Inference mode.")
    parser.add_argument("--navigation-text", default="",
                        help='[closed, live-open] Navigation instruction (1.5 / 2 only).')
    parser.add_argument("--navigation-weight", type=float, default=1.0,
                        help="[closed] Navigation CFG weight (1.5 only for CFG).")
    parser.add_argument("--vqa-question", default="",
                        help="[closed] Initial VQA question for --mode vqa.")
    parser.add_argument("--debug-worker-traceback", action="store_true",
                        help="[closed, live-open] Print async worker tracebacks on failure.")
    parser.add_argument("--record-dir", default=None,
                        help="[closed, live-open] Record camera frames, trajectory.json and "
                             "per-tick records.jsonl here; replay with --loop open --data-root.")
    return parser


def main(argv=None, preset_loop=None):
    parser = build_parser(preset_loop=preset_loop)
    args = parser.parse_args(argv)
    loop = preset_loop or args.loop

    adapter = get_adapter(args.version)

    if loop == "open":
        from module.loops import open_loop

        open_loop.run(adapter, args)
    elif loop == "closed":
        from module.loops import closed_loop

        closed_loop.run(adapter, args)
    elif loop == "live-open":
        from module.loops import live_open_loop

        live_open_loop.run(adapter, args)
    else:  # pragma: no cover - argparse restricts choices
        parser.error(f"Unknown loop {loop!r}")


if __name__ == "__main__":
    main()
