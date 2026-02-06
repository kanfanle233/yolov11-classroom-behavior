import os
import json
import time
import argparse
import subprocess
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, List

# ================= 配置区 =================
# 优先级：环境变量 > 代码硬编码
# ⚠️ 不要在代码中提交真实 Key，留空即可。
MY_API_KEY = ""


def resolve_api_key() -> Optional[str]:
    if MY_API_KEY:
        return MY_API_KEY
    return os.environ.get("DASHSCOPE_API_KEY")


def ensure_transcript_file(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.jsonl"
    if not transcript_path.exists():
        transcript_path.write_text("", encoding="utf-8")
    return transcript_path


# ================= 工具函数 =================
def extract_audio(video_path: Path, audio_out_path: Path):
    """从视频提取 16k 单声道 wav（API 常用要求）"""
    print(f"[INFO] 正在从视频提取音频...\n  视频: {video_path}\n  音频: {audio_out_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[INFO] 音频提取成功！")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 提取失败。请确认已安装 ffmpeg。\n错误信息: {e}")
        raise
    except FileNotFoundError:
        print("[ERROR] 未找到 ffmpeg 命令。请先安装 ffmpeg 并添加到环境变量。")
        raise


def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_seconds_maybe(value: Any) -> Optional[float]:
    """
    兼容：
    - 毫秒整型：1200 -> 1.2
    - 秒浮点：1.2 -> 1.2
    - 字符串数字："1200" / "1.2"
    规则：
    - 若数值 > 1000 且看起来不像秒（比如 1200/3500），按毫秒处理
    - 若在 [0, 1000] 的浮点/整型，按秒处理
    """
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None

    # 很多 ASR 用 ms，典型值上千；课堂视频秒数通常<几万秒，但 ms 更常见
    if x > 1000.0:
        return x / 1000.0
    return x


def normalize_segment(seg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    把任意 ASR sentence/segment 归一化成：
      {"start": float, "end": float, "text": str}
    """
    text = _safe_get(seg, ["text", "sentence", "content", "transcript"], "")
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    if not text:
        return None

    # 兼容 begin_time/start_time 等字段
    start_raw = _safe_get(seg, ["begin_time", "start_time", "begin", "start", "from", "offset_start"])
    end_raw = _safe_get(seg, ["end_time", "finish_time", "end", "to", "offset_end"])

    start_s = _to_seconds_maybe(start_raw)
    end_s = _to_seconds_maybe(end_raw)

    # 若没给时间戳，就跳过（否则 overlay 无法对齐）
    if start_s is None or end_s is None:
        return None

    # 修正异常：end < start
    if end_s < start_s:
        start_s, end_s = end_s, start_s

    # 修正异常：太短的片段（有些会给同一时间点）
    if end_s - start_s < 0.02:
        end_s = start_s + 0.2

    return {
        "start": round(float(start_s), 2),
        "end": round(float(end_s), 2),
        "text": text
    }


def postprocess_and_write(segments: List[Dict[str, Any]], out_path: Path):
    """
    统一后处理：
    - 按 start 排序
    - 去除明显重复/重叠的短片段
    - 写入 transcript.jsonl（统一 schema）
    """
    if not segments:
        print("[WARN] 没有可写入的 transcript 段落（segments 为空）。")
        # 仍然创建空文件，避免下游找不到
        out_path.write_text("", encoding="utf-8")
        return

    segments.sort(key=lambda x: (x["start"], x["end"]))

    cleaned: List[Dict[str, Any]] = []
    last = None
    for seg in segments:
        if last is None:
            cleaned.append(seg)
            last = seg
            continue

        # 如果文本完全相同且时间高度重叠，去重
        overlap = min(last["end"], seg["end"]) - max(last["start"], seg["start"])
        if seg["text"] == last["text"] and overlap > 0:
            # 合并成更大的区间
            last["end"] = max(last["end"], seg["end"])
            continue

        # 如果 seg 几乎完全被 last 覆盖且非常短，丢弃
        if seg["start"] >= last["start"] and seg["end"] <= last["end"] and (seg["end"] - seg["start"] < 0.3):
            continue

        cleaned.append(seg)
        last = seg

    with open(out_path, "w", encoding="utf-8") as f:
        for seg in cleaned:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    print(f"[INFO] transcript.jsonl 写入完成：{out_path}（segments={len(cleaned)}）")


# ================= DashScope Callback =================
class TranscriptCallback:
    """
    实时收集 ASR 输出，但不直接写最终文件。
    先收集 raw sentence，再统一 normalize + postprocess，保证 schema 稳定。
    """
    def __init__(self, is_sentence_end=None):
        self.raw_sentences: List[Dict[str, Any]] = []
        self._is_sentence_end = is_sentence_end

    def on_complete(self) -> None:
        print("\n[INFO] 语音识别任务流关闭。")

    def on_error(self, result) -> None:
        print(f"\n[ERROR] 识别出错: {getattr(result, 'message', 'unknown error')}")

    def on_event(self, result) -> None:
        # DashScope 的 sentence 结构可能随模型变化
        sentence = result.get_sentence()

        # 只在 sentence end 时记录（避免中间 partial）
        if self._is_sentence_end:
            try:
                is_end = self._is_sentence_end(sentence)
            except Exception:
                is_end = bool(sentence.get("text"))
        else:
            is_end = bool(sentence.get("text"))

        if not sentence or "text" not in sentence:
            return

        if is_end:
            self.raw_sentences.append(sentence)

            # 这里打印“归一化预览”，方便你肉眼确认单位是否正确
            preview = normalize_segment(sentence)
            if preview:
                print(f"[{preview['start']}s - {preview['end']}s] {preview['text']}")
            else:
                # 打印一条提示，不影响流程
                print("[WARN] 收到 sentence_end，但无法归一化（可能缺少时间戳字段或字段名变化）。")


# ================= main =================
def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="DashScope ASR 实时识别脚本（输出统一 transcript.jsonl schema）")
    parser.add_argument("--video", type=str, required=True, help="视频路径 (用于提取音频)")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")

    # 可选：指定模型（你以后换模型不用改代码）
    parser.add_argument("--asr_model", type=str, default="paraformer-realtime-v1", help="DashScope ASR model name")

    # 可选：强制重新提取音频 / 重新识别
    parser.add_argument("--force", action="store_true", help="force re-extract wav and re-run asr")

    args = parser.parse_args()

    api_key = resolve_api_key()
    if not api_key:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (base_dir / out_dir).resolve()
        transcript_path = ensure_transcript_file(out_dir)
        print("[WARN] 未配置 DashScope API Key，已生成空 transcript.jsonl 并跳过 ASR。")
        print(f"[PATH] {transcript_path}")
        return

    dashscope.api_key = api_key

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()

    api_key = resolve_api_key()
    if not api_key:
        transcript_path = ensure_transcript_file(out_dir)
        print("[WARN] 未配置 DashScope API Key，已生成空 transcript.jsonl 并跳过 ASR。")
        print(f"[PATH] {transcript_path}")
        return

    if importlib.util.find_spec("dashscope") is None:
        transcript_path = ensure_transcript_file(out_dir)
        print("[WARN] 未安装 dashscope，已生成空 transcript.jsonl 并跳过 ASR。")
        print(f"[PATH] {transcript_path}")
        return

    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

    dashscope.api_key = api_key

    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    wav_path = out_dir / "asr_audio_16k.wav"
    jsonl_path = out_dir / "transcript.jsonl"

    if not video_path.exists():
        raise FileNotFoundError(f"[ERROR] 找不到视频文件: {video_path}")

    # 需要重跑时，删除旧产物
    if args.force:
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)
        if jsonl_path.exists():
            jsonl_path.unlink(missing_ok=True)

    # wav 不存在则提取
    if not wav_path.exists():
        extract_audio(video_path, wav_path)
    else:
        print(f"[INFO] 检测到已有音频文件: {wav_path}，跳过提取。")

    class DashScopeCallback(RecognitionCallback, TranscriptCallback):
        def __init__(self, is_sentence_end=None):
            RecognitionCallback.__init__(self)
            TranscriptCallback.__init__(self, is_sentence_end=is_sentence_end)

    callback = DashScopeCallback(is_sentence_end=RecognitionResult.is_sentence_end)
    recognition = Recognition(
        model=args.asr_model,
        format="wav",
        sample_rate=16000,
        callback=callback
    )

    print(f"[INFO] 正在识别音频: {wav_path.name}... (model={args.asr_model})")
    recognition.start()

    try:
        with open(wav_path, "rb") as f:
            chunk_size = 3200
            while True:
                audio_data = f.read(chunk_size)
                if not audio_data:
                    break
                recognition.send_audio_frame(audio_data)
                time.sleep(0.05)
    except Exception as e:
        print(f"[EXCEPTION] 运行时异常: {e}")
    finally:
        recognition.stop()

    # === 统一归一化输出（关键）===
    normalized: List[Dict[str, Any]] = []
    for s in callback.raw_sentences:
        seg = normalize_segment(s)
        if seg:
            normalized.append(seg)

    postprocess_and_write(normalized, jsonl_path)
    print(f"[INFO] 识别结果已保存至: {jsonl_path}")


if __name__ == "__main__":
    main()
