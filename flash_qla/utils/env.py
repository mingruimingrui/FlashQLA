# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import tilelang
from packaging.version import Version


TILELANG_VERSION = tilelang.__version__

TILELANG_0_1_9 = Version(TILELANG_VERSION) == Version("0.1.9")
TILELANG_0_1_11 = Version(TILELANG_VERSION) == Version("0.1.11")
TILELANG_0_1_12 = Version(TILELANG_VERSION) == Version("0.1.12")
