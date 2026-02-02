import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import warnings

# 抑制 Whisper 的一些警告
warnings.filterwarnings("ignore")


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def extract_audio(video_path: Path, audio_path: Path):
    """使用 ffmpeg 从视频提取音频 (wav)"""
    if audio_path.exists():
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # whisper 需要 16k
        "-ac", "1",
        str(audio_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def run_whisper(video_path: str, out_dir: str, model_size: str = "base"):
    video_p = Path(video_path)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    audio_p = out_p / f"{video_p.stem}.wav"
    jsonl_p = out_p / "transcript.jsonl"

    if jsonl_p.exists() and jsonl_p.stat().st_size > 0:
        print(f"[ASR] Skip existing: {jsonl_p}")
        return

    if not check_ffmpeg():
        print("[Error] ffmpeg not found. Please install ffmpeg to extract audio.")
        sys.exit(1)

    print(f"[ASR] Extracting audio from {video_p.name}...")
    try:
        extract_audio(video_p, audio_p)
    except Exception as e:
        print(f"[Error] Audio extraction failed: {e}")
        return

    print(f"[ASR] Loading Whisper model ({model_size})...")
    try:
        import whisper
    except ImportError:
        print("[Error] 'openai-whisper' not installed. Run: pip install openai-whisper")
        sys.exit(1)

    # 加载模型 (自动利用 GPU)
    model = whisper.load_model(model_size)

    print(f"[ASR] Transcribing...")
    # beam_size=5 提升准确率
    result = model.transcribe(str(audio_p), beam_size=5, fp16=False, language="zh")

    segments = result.get("segments", [])

    print(f"[ASR] Writing {len(segments)} segments to {jsonl_p.name}...")

    with open(jsonl_p, "w", encoding="utf-8") as f:
        for seg in segments:
            record = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
                "conf": float(seg.get("avg_logprob", 0.0))  # 粗略置信度
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 清理临时音频文件 (可选)
    # audio_p.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="base", help="tiny, base, small, medium, large")
    args = parser.parse_args()

    run_whisper(args.video, args.out_dir, args.model)