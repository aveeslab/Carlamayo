# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Backward-compatible open-loop wrapper around the unified carlamayo launcher.

"""Open-loop entrypoint. Equivalent to ``carlamayo.py --loop open``.

Requires ``--version {1,1.5,2,2-edge}``; see ``python carlamayo.py --loop open --help``.
"""

from carlamayo import main

if __name__ == "__main__":
    main(preset_loop="open")
