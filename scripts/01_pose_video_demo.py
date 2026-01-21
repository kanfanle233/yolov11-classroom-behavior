import os
import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path


def main():
    # ====== 0) 项目根目录（YOLOv11） ======
    project_root = Path(__file__).resolve().parents[1]

    # ====== 1) 命令行参数（默认仍然跑 demo1） ======
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/videos/demo1.mp4", help="relative path to project root")
    parser.add_argument("--out", type=str, default="output/pose_demo_out.mp4", help="relative path to project root")
    # ✅ 修复：默认模型放在项目根目录，而不是 runs/
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="relative path to project root")

    args = parser.parse_args()

    video_path = str((project_root / args.video).resolve())
    out_path = str((project_root / args.out).resolve())
    model_path = str((project_root / args.model).resolve())

    output_dir = str(Path(out_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    # ====== 2) 先检查文件是否存在 ======
    print("[CHECK] video exists?", os.path.exists(video_path), video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print("[CHECK] model exists?", os.path.exists(model_path), model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ====== 3) 加载 YOLO11 Pose 预训练模型 ======
    model = YOLO(model_path)

    # ====== 4) 打开视频 ======
    cap = cv2.VideoCapture(video_path)
    print("[CHECK] cap opened?", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV cannot open this video. "
            "可能是编码(HEVC/H.265)或OpenCV缺少解码器。"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ====== 5) 输出视频写入器 ======
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter open failed. 可以尝试改成输出 .avi 或换 fourcc。")

    print(f"[INFO] Input : {video_path}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Model : {model_path}")
    print(f"[INFO] FPS={fps:.2f}, Size={w}x{h}")

    # ====== 6) 逐帧推理 ======
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            verbose=False,
            device=0,  # 强制用第一块GPU
            half=True,  # FP16，4060 非常吃香（显存/速度直接改善）
            imgsz=960,  # 1080p 建议 960 或 832；想更快用 640
            conf=0.25
        )

        annotated = results[0].plot()
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO] processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print("[DONE] Saved:", out_path)


if __name__ == "__main__":
    main()
