#!/usr/bin/env python3

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve EasyR1 checkpoint paths from checkpoint_tracker.json.")
    parser.add_argument("--checkpoint_root", required=True, help="Checkpoint directory for one experiment.")
    parser.add_argument("--selection", choices=("last", "best"), default="best")
    parser.add_argument("--artifact", choices=("global_step", "actor"), default="actor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker_path = Path(args.checkpoint_root) / "checkpoint_tracker.json"
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))

    if args.selection == "last":
        global_step = tracker["last_global_step"]
    else:
        global_step = tracker["best_global_step"]

    step_dir = Path(args.checkpoint_root) / f"global_step_{global_step}"
    if args.artifact == "global_step":
        print(step_dir)
    else:
        print(step_dir / "actor")


if __name__ == "__main__":
    main()