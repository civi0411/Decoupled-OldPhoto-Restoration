from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gradio_demo import launch_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Gradio demo for the auto-mask restoration pipeline.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    launch_app(server_name=args.server_name, server_port=args.server_port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
