"""把 commit_trace 的轨迹嵌进回放模板，产出单文件 HTML（可直接双击打开）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "web" / "commit_replay_template.html"
TRACE = Path(__file__).resolve().parent.parent / "cache" / "commit_trace.json"
OUT = Path(__file__).resolve().parent.parent / "docs" / "commit_replay.html"


def build(trace: Path, out: Path) -> Path:
    template = TEMPLATE.read_text(encoding="utf-8")
    if "__DATA__" not in template:
        raise SystemExit(f"{TEMPLATE} 缺少 __DATA__ 占位符")
    data = json.loads(trace.read_text(encoding="utf-8"))
    html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if out.suffix == ".html" and out.parent.name == "docs":
        (out.parent / ".nojekyll").touch()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=TRACE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = build(args.trace, args.out)
    print(f"[commit_view] {args.trace.stat().st_size/1024:.0f} KB trace -> {out} "
          f"({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
