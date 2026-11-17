"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

# In this (mainland-China) environment huggingface.co is not reachable while
# the hf-mirror.com mirror is. transformers/huggingface_hub read this variable
# at import time, so it must be set before any transformers import below.
# Set HF_ENDPOINT externally to override (e.g. back to https://huggingface.co).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from omegaconf import OmegaConf

from lavis.common.registry import registry

# NOTE: BC-HOI only uses BLIP-2 for feature extraction. Importing the full
# LAVIS tree (datasets builders / tasks / video, audio, diffusion models)
# drags in a large number of optional dependencies (iopath, decord, spacy,
# pycocoevalcap, ulip, ...), so the imports are narrowed to what is used.
from lavis.models import *
from lavis.processors import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
