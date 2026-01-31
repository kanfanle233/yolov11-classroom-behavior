import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import List

# ==========================================
# 关键修改 1：设置 Hugging Face 国内镜像，解决 SSLError 下载失败问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ==========================================

def run(cmd: List[str]) -> None:
    # 屏蔽 ffmpeg 输出
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_wav(ffmpeg_exe: str, video_path: str, wav_path: str, sr: int = 16000) -> None:
    # 提取音频
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        wav_path
    ]
    run(cmd)


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=str(base_dir / "data" / "videos" / "demo1.mp4"),
                        help="input mp4 path (absolute or relative to project root)")
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "output"),
                        help="output dir for wav + transcript.jsonl")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--sr", type=int, default=16000)

    # ==========================================
    # 关键修改 2：默认语言改为 "zh" (中文)，否则中文视频会识别成乱码
    parser.add_argument("--lang", type=str, default="zh")
    # ==========================================

    args = parser.parse_args()

    # 兼容：允许传相对路径
    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = str((base_dir / video_path).resolve())

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_dir / "asr_audio_16k.wav")
    jsonl_path = str(out_dir / "transcript.jsonl")
    ffmpeg_exe = args.ffmpeg

    if not os.path.exists(video_path):
        print(f"[ERROR] 找不到视频: {video_path}")
        return

    print("[INFO] 1. 提取音频...")
    extract_wav(ffmpeg_exe, video_path, wav_path, sr=args.sr)

    print("[INFO] 2. 加载 Whisper (CPU/Int8)...")
    try:
        from faster_whisper import WhisperModel
        # 如果 small 还是效果不好，可以尝试改用 "medium" (速度会慢一倍，但抗噪能力强很多)
        model = WhisperModel("small", device="cpu", compute_type="int8")
    except ImportError:
        print("请 pip install faster-whisper")
        return

    print(f"[INFO] 3. 开始识别 (语言={args.lang}, 已关闭 VAD 过滤器)...")

    segments, info = model.transcribe(
        wav_path,
        language=args.lang,  # 使用修改后的中文设置
        vad_filter=False,  # 关键！关闭静音过滤
        beam_size=5,
        initial_prompt=None,  # 移除提示词，避免幻觉
        condition_on_previous_text=False,  # 防止重复循环
        no_speech_threshold=0.6  # 放宽“非语音”的判定阈值
    )

    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        print("-" * 50)
        for seg in segments:
            text = seg.text.strip()
            # 过滤掉那种极短的杂音（比如 "hmm", "."）
            if len(text) < 1:  # 中文可能只有1个字，改宽一点
                continue

            obj = {
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": text
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"[{obj['start']:>6.2f}s - {obj['end']:>6.2f}s] {text}")
            n += 1

    print("-" * 50)
    print(f"[DONE] 成功识别出 {n} 条语句")
    print(f"[PATH] {jsonl_path}")

    if n > 0:
        print("✅ 成功！现在可以进入下一步（生成 per_person_sequences + overlay）。")
    else:
        print("❌ 依然为 0。可能需要检查是否需要 'volume=10dB' 的放大处理。")


if __name__ == "__main__":
    main()
