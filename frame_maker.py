from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes or return b'' on EOF."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        b = stream.read(remaining)
        if not b:
            return b""
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


@dataclass(frozen=True)
class CaptureConfig:
    gdigrab_input: str = "desktop"  # or: title=YourWindowTitle
    fps: float = 20.0
    width: int = 640
    height: int = 360


def start_ffmpeg_gdigrab_capture(cfg: CaptureConfig) -> subprocess.Popen[bytes]:
    """
    ffmpeg(gdigrab) -> stdout(rawvideo rgb24)
    -pix_fmt rgb24 로 RGB 프레임이 (H,W,3)로 들어옵니다.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "gdigrab",
        "-framerate",
        str(cfg.fps),
        "-i",
        cfg.gdigrab_input,
        "-vf",
        f"scale={cfg.width}:{cfg.height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )


def iter_frame_batches(
    *,
    cfg: CaptureConfig,
    batch: int,
    queue_max: int = 120,
    drop_oldest_when_full: bool = True,
) -> Iterator[List[np.ndarray]]:
    """
    frames 배치(list[np.ndarray RGB])를 계속 생성해서 yield 합니다.
    - “파일 저장” 없이 ffmpeg stdout에서 raw 프레임을 읽습니다.
    - batch개 프레임이 모일 때마다 yield 합니다.
    """
    if cfg.width <= 0 or cfg.height <= 0:
        raise ValueError("width/height must be positive")
    if cfg.fps <= 0:
        raise ValueError("fps must be > 0")
    if batch <= 0:
        raise ValueError("batch must be > 0")

    frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(queue_max))
    stop = threading.Event()

    def _capture_loop() -> None:
        proc = start_ffmpeg_gdigrab_capture(cfg)
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout is None")

        frame_bytes = int(cfg.width * cfg.height * 3)

        while not stop.is_set():
            raw = _read_exact(proc.stdout, frame_bytes)
            if not raw:
                time.sleep(0.01)
                continue

            frame_rgb = np.frombuffer(raw, dtype=np.uint8).reshape((cfg.height, cfg.width, 3))

            try:
                frame_q.put_nowait(frame_rgb)
            except queue.Full:
                if not drop_oldest_when_full:
                    continue
        
                try:
                    _ = frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_q.put_nowait(frame_rgb)
                except queue.Full:
                    pass

        try:
            proc.terminate()
        except Exception:
            pass

    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    buf: list[np.ndarray] = []
    try:
        while True:
            frame = frame_q.get()  
            buf.append(frame)
            if len(buf) >= batch:
                frames = buf[:batch]
                buf = []
                yield frames
    finally:
        stop.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdigrab-input", default="desktop")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--queue-max", type=int, default=120)

    # 테스트용(원하면 끄고, 다른 코드에서 iter_frame_batches만 import해서 쓰면 됨)
    ap.add_argument("--print-only", action="store_true", help="prints batch info only")

    args = ap.parse_args()

    cfg = CaptureConfig(
        gdigrab_input=str(args.gdigrab_input),
        fps=float(args.fps),
        width=int(args.width),
        height=int(args.height),
    )

    t0 = time.time()
    n = 0
    for frames in iter_frame_batches(cfg=cfg, batch=int(args.batch), queue_max=int(args.queue_max)):
        n += 1
        dt = time.time() - t0
        approx_secs = len(frames) / max(cfg.fps, 1e-6)
        print(f"[batch] n={n}  frames={len(frames)}  approx_window={approx_secs:.3f}s  elapsed={dt:.1f}s")

        if not args.print_only:
            # 여기서 frames를 원하는 모델(SAM3 등)에 넘기면 됩니다.
            # 예: masks = your_model(frames)
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())