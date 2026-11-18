"""
autoclip.py - give it a stream/VOD link, get back vertical (9:16) clips
with burned-in captions, cut from the most exciting part.

"Most exciting" = the loudest, most dynamic stretch of audio (shouting,
reactions, hype moments). It's a good proxy, not magic.

Needs: ffmpeg, yt-dlp, streamlink, faster-whisper, numpy
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SR = 16000
OUT_DIR = Path("output")
COOKIES = Path("cookies.txt")  # optional, helps with YouTube


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def cookie_args():
    args = ["--js-runtimes", "node"]
    if COOKIES.exists():
        args += ["--cookies", str(COOKIES)]
    return args


# ---------------------------------------------------------------- fetching

def is_live(url):
    try:
        out = subprocess.run(
            ["yt-dlp", "--print", "%(is_live)s", "--no-warnings", *cookie_args(), url],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip().splitlines()
        return bool(out) and out[-1].strip() == "True"
    except Exception:
        return False


def record_live(url, minutes, wd):
    """Record the live stream for N minutes into a single .ts file."""
    dest = wd / "live.ts"
    print(f"Stream is live. Recording {minutes} minutes...", flush=True)
    subprocess.run([
        "timeout", str(int(minutes * 60)),
        "streamlink", url, "720p60,720p,best",
        "-o", str(dest), "--force", "--quiet",
    ])  # timeout exits non-zero on purpose, so no check=True
    if not dest.exists() or dest.stat().st_size == 0:
        sys.exit("Recording failed (is the stream actually live?).")
    return dest


def fetch_audio(url, wd):
    run(["yt-dlp", "-f", "ba/worst", "-o", str(wd / "audio.%(ext)s"),
         *cookie_args(), url])
    files = list(wd.glob("audio.*"))
    if not files:
        sys.exit("Could not download audio from that link.")
    return files[0]


# ---------------------------------------------------------- finding moments

def loudness_db(path):
    """Loudness in dB for every second of audio (streamed, low memory)."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    vals = []
    chunk = SR * 2  # one second of 16-bit mono
    while True:
        buf = proc.stdout.read(chunk)
        if len(buf) < chunk:
            break
        a = np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
        vals.append(20 * np.log10(np.sqrt(float(np.mean(a * a))) + 1e-6))
    proc.wait()
    return np.array(vals)


def pick_windows(db, length, count):
    """Return start seconds of the best non-overlapping windows."""
    n = len(db)
    k = int(length)
    if n <= k:
        return [0]

    floor = np.percentile(db, 10)
    x = np.clip(db - floor, 0, None)
    x = x / (x.max() + 1e-6)

    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    var = np.clip((c2[k:] - c2[:-k]) / k - mean ** 2, 0, None)
    scores = mean + 0.5 * np.sqrt(var)  # loud AND dynamic

    if len(scores) > 60:
        scores[:30] = -1  # skip stream intros

    picks = []
    for _ in range(count):
        i = int(np.argmax(scores))
        if scores[i] < 0:
            break
        picks.append(i)
        scores[max(0, i - k): i + k] = -1
    return picks


# ------------------------------------------------------------------ captions

def ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,DejaVu Sans,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,7,0,2,60,60,430,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def make_captions(clip, ass_path, model_size):
    """Transcribe with Whisper, write 3-word-at-a-time ASS captions."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(clip), word_timestamps=True, vad_filter=True)
    words = [w for seg in segments for w in (seg.words or [])]

    lines = []
    for i in range(0, len(words), 3):
        group = words[i:i + 3]
        text = " ".join(w.word.strip() for w in group).upper()
        text = re.sub(r"[{}\\]", "", text)
        if not text.strip():
            continue
        start = group[0].start
        end = group[-1].end
        if i + 3 < len(words):
            end = max(end, min(words[i + 3].start, end + 0.4))
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end + 0.1)},Default,,0,0,0,,{text}"
        )

    ass_path.write_text(ASS_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# ----------------------------------------------------------------- rendering

def cut_raw(url, live_file, start, length, wd, idx, height=1080):
    raw = wd / f"raw_{idx}.mp4"
    if live_file:
        run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(live_file),
             "-t", str(length), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "aac", str(raw)])
    else:
        run(["yt-dlp", "-f",
             f"bv*[height<={height}]+ba/b[height<={height}]/b",
             "--download-sections", f"*{start}-{start + length}",
             "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
             "-o", str(raw), *cookie_args(), url])
    if not raw.exists():
        sys.exit("Could not cut the clip from the source.")
    return raw


def render_vertical(raw, ass_path, has_captions, out, framing, wd):
    if framing == "crop":
        base = "[0:v]scale=-2:1920,crop=1080:1920[v]"
    else:  # blur: whole frame kept, blurred copy fills the background
        base = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=25:3[b];"
            "[fg]scale=1080:-2[f];"
            "[b][f]overlay=(W-w)/2:(H-h)/2[v]"
        )
    graph = base + (";[v]subtitles=cap.ass[outv]" if has_captions else ";[v]null[outv]")

    run(["ffmpeg", "-v", "error", "-y", "-i", str(raw),
         "-filter_complex", graph, "-map", "[outv]", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", str(out)],
        cwd=str(wd))


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--length", type=int, default=30, help="clip length in seconds")
    p.add_argument("--count", type=int, default=1, help="how many clips")
    p.add_argument("--record-minutes", type=float, default=20,
                   help="if the stream is live, how long to record first")
    p.add_argument("--framing", choices=["blur", "crop"], default="blur")
    p.add_argument("--whisper-model", default="base")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)

        live_file = None
        if is_live(args.url):
            live_file = record_live(args.url, args.record_minutes, wd)
            audio = live_file
        else:
            print("Not live, treating it as a VOD.", flush=True)
            audio = fetch_audio(args.url, wd)

        print("Scanning for the best moments...", flush=True)
        db = loudness_db(audio)
        if len(db) < 5:
            sys.exit("Not enough audio to analyze.")
        starts = pick_windows(db, args.length, args.count)

        for n, start in enumerate(starts, 1):
            print(f"Clip {n}: starting at {start}s", flush=True)
            raw = cut_raw(args.url, live_file, start, args.length, wd, n)
            ass = wd / "cap.ass"
            count = make_captions(raw, ass, args.whisper_model)
            out = OUT_DIR / f"clip_{n}_at_{start}s.mp4"
            render_vertical(raw, ass, count > 0, out, args.framing, wd)
            print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
  
