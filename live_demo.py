from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterable

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


def _color_for_id(obj_id: int) -> np.ndarray:
    base = (obj_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 60 + (base % 196)
    g = 60 + ((base // 7) % 196)
    b = 60 + ((base // 49) % 196)
    return np.array([r, g, b], dtype=np.float32)  


def overlay_instance_masks_rgb(
    frame_rgb: np.ndarray,
    *,
    masks: np.ndarray | None,
    object_ids: Iterable[int] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    if masks is None:
        return frame_rgb
    masks_np = np.asarray(masks)
    if masks_np.ndim != 3:
        return frame_rgb

    out = frame_rgb.astype(np.float32).copy()

    if object_ids is None:
        obj_ids = list(range(int(masks_np.shape[0])))
    else:
        obj_ids = [int(x) for x in list(object_ids)]
        if len(obj_ids) != int(masks_np.shape[0]):
            obj_ids = list(range(int(masks_np.shape[0])))

    for i in range(int(masks_np.shape[0])):
        m = masks_np[i] > 0
        if not bool(np.any(m)):
            continue
        color = _color_for_id(int(obj_ids[i]))
        out[m] = alpha * color + (1.0 - alpha) * out[m]

    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class CaptureConfig:
    gdigrab_input: str
    fps: float
    width: int
    height: int

#프레임 캡쳐 후 메모리에 넣기
def start_ffmpeg_gdigrab_capture(cfg: CaptureConfig) -> subprocess.Popen[bytes]:
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
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)


def run_sam3_on_frames(
    *,
    frames_rgb: list[np.ndarray],
    prompts: list[str],
    device: str,
    processing_device: str,
    video_storage_device: str,
    model_name: str,
    dtype: str,
) -> tuple[np.ndarray | None, list[int] | None]:
    import torch
    from accelerate import Accelerator
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    accel_device = Accelerator().device
    if device == "cuda" and str(accel_device) == "cpu":
        print("[warn] --device cuda requested, but Accelerator device is cpu. Running on CPU.")

    model = Sam3VideoModel.from_pretrained(model_name).to(accel_device, dtype=torch_dtype)
    processor = Sam3VideoProcessor.from_pretrained(model_name)

    session = processor.init_video_session(
        video=frames_rgb,
        inference_device=accel_device,
        processing_device=processing_device,
        video_storage_device=video_storage_device,
        dtype=torch_dtype,
    )
    session = processor.add_text_prompt(
        inference_session=session, text=prompts if len(prompts) > 1 else prompts[0]
    )

    last_processed = None
    with torch.inference_mode():
        for mo in model.propagate_in_video_iterator(
            inference_session=session, max_frame_num_to_track=len(frames_rgb) - 1
        ):
            last_processed = processor.postprocess_outputs(session, mo)

    if last_processed is None:
        return None, None

    masks = last_processed.get("masks", None)
    obj_ids = last_processed.get("object_ids", None)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    if hasattr(obj_ids, "detach"):
        obj_ids = obj_ids.detach().cpu().numpy()
    obj_ids_list = None
    if obj_ids is not None:
        obj_ids_list = [int(x) for x in np.asarray(obj_ids).tolist()]

    return (np.asarray(masks) if masks is not None else None), obj_ids_list


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gdigrab-input",
        default="desktop",
        help='ffmpeg gdigrab input. Default "desktop". You can also try: title=YourWindowTitle',
    )
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--batch", type=int, default=16, help="number of frames per SAM3 run")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--model-name", default="facebook/sam3")
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["objects in refrigerator"],
        help='one or more prompts, e.g. --prompts "objects in refrigerator" hands',
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--processing-device", default=None)
    ap.add_argument("--video-storage-device", default=None)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    ap.add_argument("--queue-max", type=int, default=120, help="max frames buffered in memory")
    args = ap.parse_args()

    import cv2

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width/--height must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.batch <= 0:
        raise SystemExit("--batch must be > 0")

    processing_device = args.processing_device or args.device
    video_storage_device = args.video_storage_device or args.device

   
    if args.dtype is not None:
        dtype = args.dtype
    else:
        dtype = "float32" if args.device == "cpu" else "bfloat16"

    frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(args.queue_max))
    stop = threading.Event()

    def _capture_loop() -> None:
        cfg = CaptureConfig(
            gdigrab_input=str(args.gdigrab_input),
            fps=float(args.fps),
            width=int(args.width),
            height=int(args.height),
        )
        proc = start_ffmpeg_gdigrab_capture(cfg)
        if proc.stdout is None:
            print("[error] ffmpeg stdout is None")
            return

        frame_bytes = int(cfg.width * cfg.height * 3)
        while not stop.is_set():
            raw = _read_exact(proc.stdout, frame_bytes)
            if not raw:
                time.sleep(0.05)
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((cfg.height, cfg.width, 3))
            
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                try:
                    _ = frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_q.put_nowait(frame)
                except queue.Full:
                    pass

        try:
            proc.terminate()
        except Exception:
            pass

    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    cv2.namedWindow("SAM3 live (micro-batch)", cv2.WINDOW_NORMAL)

    buf: list[np.ndarray] = []
    last_overlay: np.ndarray | None = None
    last_infer_ms: float | None = None
    last_n_obj: int | None = None

    try:
        while True:
            try:
                frame = frame_q.get(timeout=0.25)
            except queue.Empty:
                continue

            buf.append(frame)
            display = last_overlay if last_overlay is not None else frame

            # Draw HUD
            bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            hud = f"fps={args.fps:g}  batch={args.batch}  device={args.device}  prompt={args.prompts[0]}"
            cv2.putText(
                bgr,
                hud,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if last_infer_ms is not None:
                hud2 = f"infer={last_infer_ms:.0f}ms  n_obj={last_n_obj or 0}"
                cv2.putText(
                    bgr,
                    hud2,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("SAM3 live (micro-batch)", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if len(buf) < int(args.batch):
                continue

            frames = buf[: int(args.batch)]
            buf = []

            t0 = time.time()
            try:
                masks, obj_ids = run_sam3_on_frames(
                    frames_rgb=frames,
                    prompts=[str(p) for p in args.prompts],
                    device=str(args.device),
                    processing_device=str(processing_device),
                    video_storage_device=str(video_storage_device),
                    model_name=str(args.model_name),
                    dtype=str(dtype),
                )
            except Exception as e:
                print("[error] SAM3 inference failed:", repr(e))
                masks, obj_ids = None, None
            dt_ms = (time.time() - t0) * 1000.0
            last_infer_ms = float(dt_ms)

            if masks is not None:
                last_n_obj = int(np.asarray(masks).shape[0])
                last_overlay = overlay_instance_masks_rgb(
                    frames[-1],
                    masks=masks,
                    object_ids=obj_ids,
                    alpha=float(args.alpha),
                )
            else:
                last_n_obj = 0
                last_overlay = frames[-1]

    finally:
        stop.set()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


