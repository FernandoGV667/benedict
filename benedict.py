#!/usr/bin/env python3
import sys
import os
import re
import html
import math
import argparse
import urllib.request
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache

# Early environment setup
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---- Constants ----
HELPER_FLAGS = frozenset({
    "-c", "-B", "-S", "-I",
    "--multiprocessing-fork", "--multiprocessing-spawn", "--multiprocessing-forkserver",
})
DEFAULT_LANGS = ("en", "en-US", "en-GB", "es", "es-419", "es-ES")
AUDIO_SEGMENT_DURATION = 30  # seconds
SAMPLE_RATE = 16000
CAPTION_FORMATS = {"vtt": 0, "srt": 1}

# ---- Helper-process guard for frozen builds (PyInstaller)
if getattr(sys, "frozen", False) and any(arg in HELPER_FLAGS for arg in sys.argv[1:]):
    sys.exit(0)

# --- Imports after guard (lazy loading for better startup)
import numpy as np
import librosa
import yt_dlp
from tqdm import tqdm
import torch
import whisper
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# Compile regex patterns once
VIDEO_ID_PATTERN = re.compile(r"[?&]v=([^&]+)")
TIMESTAMP_PATTERN = re.compile(r"^\d{1,2}:\d{2}:\d{2}[\.,]\d{3}\s+--\>\s+\d{1,2}:\d{2}:\d{2}[\.,]\d{3}")
SRT_INDEX_PATTERN = re.compile(r"^\d+$")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

# ------------------ YT captions first ------------------
@lru_cache(maxsize=128)
def _extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL with caching."""
    if "youtu.be/" in url:
        return url.split("youtu.be/", 1)[1].split("?", 1)[0]
    match = VIDEO_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid YouTube URL: cannot extract video ID from {url}")

def _entries_to_text(entries) -> str:
    """Normalize entries to plain text efficiently."""
    if hasattr(entries, "to_raw_data"):
        entries = entries.to_raw_data()
    
    out = []
    append_out = out.append  # Local reference for speed
    
    for e in entries:
        if isinstance(e, dict):
            line = (e.get("text") or "").strip()
        else:
            line = getattr(e, "text", str(e)).strip()
        if line:
            append_out(html.unescape(line))
    
    return MULTI_NEWLINE_PATTERN.sub("\n\n", "\n".join(out).strip())

def try_fetch_youtube_transcript(
    video_url: str, 
    preferred_langs: Optional[Tuple[str, ...]] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch YouTube transcript with optimized error handling.
    Returns (text, lang, source) or (None, None, None).
    """
    if preferred_langs is None:
        preferred_langs = DEFAULT_LANGS

    try:
        vid = _extract_video_id(video_url)
        api = YouTubeTranscriptApi()
        listing = api.list(vid)

        # Try to fetch transcripts in order of preference
        for is_manual in (True, False):
            for lang in preferred_langs:
                try:
                    t = (listing.find_manually_created_transcript([lang]) if is_manual 
                         else listing.find_generated_transcript([lang]))
                    fetched = t.fetch()
                    txt = _entries_to_text(fetched)
                    if txt:
                        lang_code = getattr(fetched, "language_code", lang)
                        return txt, lang_code, "manual" if is_manual else "auto"
                except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
                    continue
                except Exception:
                    continue

        # Fallback: Let API choose from preferred langs
        try:
            fetched = api.fetch(vid, languages=preferred_langs)
            txt = _entries_to_text(fetched)
            if txt:
                lang_code = getattr(fetched, "language_code", preferred_langs[0])
                source = "auto" if getattr(fetched, "is_generated", False) else "manual"
                return txt, lang_code, source
        except Exception:
            pass

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, ValueError):
        pass
    
    return None, None, None

# ------------------ yt_dlp captions fallback ------------------
def _choose_caption_track(
    tracks_by_lang: Dict[str, List[Dict]], 
    preferred_langs: Tuple[str, ...]
) -> Tuple[Optional[str], Optional[Dict]]:
    """Optimized caption track selection."""
    if not tracks_by_lang:
        return None, None
    
    # Build a lookup for faster access
    def get_best_track(tracks):
        return min(tracks, key=lambda t: CAPTION_FORMATS.get(t.get("ext", ""), 9))
    
    # Exact match
    for lang in preferred_langs:
        if lang in tracks_by_lang and tracks_by_lang[lang]:
            return lang, get_best_track(tracks_by_lang[lang])
    
    # Prefix match
    for lang in preferred_langs:
        for key, tracks in tracks_by_lang.items():
            if key.startswith(lang) and tracks:
                return key, get_best_track(tracks)
    
    # Any available
    for key, tracks in tracks_by_lang.items():
        if tracks:
            return key, get_best_track(tracks)
    
    return None, None

def _download_text(url: str) -> str:
    """Download and decode text with proper encoding handling."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    
    # Try UTF-8 first (most common), then fallback
    for encoding in ('utf-8', 'latin-1'):
        try:
            return data.decode(encoding, errors='strict' if encoding == 'utf-8' else 'ignore')
        except UnicodeDecodeError:
            if encoding == 'latin-1':
                raise

def _strip_timed_captions(payload: str) -> str:
    """Remove timing/index lines from WebVTT/SRT efficiently."""
    lines = payload.splitlines()
    out = []
    append_out = out.append
    
    for line in lines:
        s = line.strip()
        if not s:
            append_out(line)
            continue
        
        # Skip timing and format lines
        if (s.upper().startswith("WEBVTT") or 
            TIMESTAMP_PATTERN.match(s) or 
            SRT_INDEX_PATTERN.match(s) or 
            "-->" in s):
            continue
        
        append_out(line)
    
    return MULTI_NEWLINE_PATTERN.sub("\n\n", "\n".join(out).strip())

def try_fetch_ytdlp_captions(
    url: str, 
    preferred_langs: Optional[Tuple[str, ...]] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch captions via yt_dlp with optimized settings."""
    if preferred_langs is None:
        preferred_langs = DEFAULT_LANGS

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt/srt",
        "nocheckcertificate": True,
    }
    
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None, None, None

    # Try manual first, then auto
    for captions, source in [(info.get("subtitles", {}), "manual"), 
                              (info.get("automatic_captions", {}), "auto")]:
        lang, track = _choose_caption_track(captions, preferred_langs)
        if track and track.get("url"):
            try:
                payload = _download_text(track["url"])
                text = _strip_timed_captions(payload)
                if text.strip():
                    return text, lang, source
            except Exception:
                continue
    
    return None, None, None

# ------------------ Whisper fallback ------------------
@lru_cache(maxsize=1)
def _is_mps_usable() -> bool:
    """Check MPS availability with caching."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.randn(8, device="mps", dtype=torch.float32)
        _ = (x * 2).relu().sum().item()
        return True
    except Exception:
        return False

def download_audio(
    youtube_url: str, 
    output_dir: str,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None
) -> Tuple[str, Optional[float]]:
    """Download audio with optimized format selection."""
    
    base_opts = {
        'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'hls_prefer_native': True,
        'extractor_args': {'youtube': {'player_client': ['ios', 'web']}},
    }
    
    if cookies_from_browser:
        base_opts['cookiesfrombrowser'] = (cookies_from_browser,)
    if cookies_file:
        base_opts['cookiefile'] = cookies_file

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
    
    formats = info.get('formats', [])
    
    # Optimized format selection using generators
    def score(f):
        return f.get('abr') or f.get('tbr') or 0
    
    def is_compatible_format(f):
        proto = (f.get('protocol') or '').lower()
        ext = (f.get('ext') or '').lower()
        acodec = (f.get('acodec') or '').lower()
        vcodec = f.get('vcodec')
        
        # Progressive MP4 with AAC
        if vcodec != 'none' and proto in ('http', 'https') and ext == 'mp4':
            return acodec.startswith('mp4a') or acodec in ('aac', 'unknown')
        
        # Audio-only AAC/M4A
        if vcodec == 'none':
            if 'm3u8' in proto and ext == 'mp4':
                return True
            if 'http' in proto and ext in ('m4a', 'mp4'):
                return acodec.startswith('mp4a') or acodec == 'aac'
        
        return False
    
    # Find best compatible format
    compatible = [f for f in formats if is_compatible_format(f)]
    
    if compatible:
        # Prefer format 18 if available, otherwise highest quality
        by_id = {f.get('format_id'): f for f in compatible}
        chosen = by_id.get('18') or max(compatible, key=score)
        dl_opts = {**base_opts, 'format': chosen['format_id']}
    else:
        # Fallback format expression
        dl_opts = {**base_opts, 
                   'format': "18/bestaudio[ext=mp4][protocol!=m3u8]/bestaudio[acodec^=mp4a]"}
    
    # Download with error handling
    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            info2 = ydl.extract_info(youtube_url, download=True)
    except yt_dlp.utils.DownloadError:
        # Robust fallback
        dl_opts['format'] = "18/bestaudio[ext=mp4]/bestaudio"
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            info2 = ydl.extract_info(youtube_url, download=True)
    
    ext = info2.get('ext', 'mp4')
    return os.path.join(output_dir, f'audio.{ext}'), info.get('duration')

def select_model(duration_sec: Optional[float]) -> str:
    """Select Whisper model based on duration."""
    if duration_sec is None:
        return "base"
    
    # Use tuple for cleaner threshold checking
    thresholds = ((600, "small"), (1800, "medium"))
    for threshold, model in thresholds:
        if duration_sec < threshold:
            return model
    return "large"

def load_audio_no_ffmpeg(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio without ffmpeg dependency."""
    import audioread
    
    with audioread.audio_open(path) as f:
        in_sr = f.samplerate
        n_channels = f.channels
        # Read all chunks at once
        raw = b"".join(f.read_data())
    
    # Convert to float32 in single operation
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32, copy=False) / 32768.0
    
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    
    if in_sr != sr:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sr)
    
    return audio

def transcribe_with_whisper(
    audio_path: str, 
    model_size: str,
    forced_device: Optional[str] = None
) -> Tuple[str, str]:
    """Transcribe audio using Whisper with optimized processing."""
    
    # Device selection
    if forced_device:
        device = forced_device
    else:
        if torch.cuda.is_available():
            device = "cuda"
            print("[Benedict] Using NVIDIA GPU (CUDA)")
        elif _is_mps_usable():
            device = "mps"
            print("[Benedict] Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            print("[Benedict] No usable GPU found. Using CPU")
    
    # Load model on CPU first
    model = whisper.load_model(model_size, device="cpu")
    
    # Move to target device - MPS and CUDA need different handling
    if device == "mps":
        try:
            model = model.to("mps", dtype=torch.float32)
        except Exception:
            # If MPS fails at this point, quietly continue on CPU
            device = "cpu"
            model = model.to("cpu")
    elif device == "cuda":
        try:
            model = model.to("cuda")
        except Exception as e:
            print(f"CUDA move failed ({type(e).__name__}). Using CPU.")
            device = "cpu"
            model = model.to("cpu")
    # device == "cpu" already handled by default
    
    # Get model configuration
    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    model_device = first_param.device
    n_mels = model.dims.n_mels
    
    print(f"[Benedict] Using Whisper model: {model_size} on {model_device.type}")
    print(f"[Benedict] Model expects {n_mels} mel channels")
    
    # Load audio
    audio = load_audio_no_ffmpeg(audio_path, sr=SAMPLE_RATE)
    samples_per_segment = AUDIO_SEGMENT_DURATION * SAMPLE_RATE
    total_segments = math.ceil(len(audio) / samples_per_segment)
    
    print(f"[Benedict] Total segments: {total_segments}")
    
    # Detect language once (sample from the middle to avoid intro music/silence)
    mid_start = max(0, (len(audio) - samples_per_segment) // 2)
    sample_segment = whisper.pad_or_trim(audio[mid_start:mid_start + samples_per_segment])
    mel = whisper.log_mel_spectrogram(sample_segment, n_mels=n_mels).to(model_device, dtype=model_dtype)
    
    _, lang_probs = model.detect_language(mel)
    detected_lang = max(lang_probs, key=lang_probs.get)
    top3 = sorted(lang_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"[Benedict] Detected language: {detected_lang} (top: {top3})")
    
    # Transcribe with pre-allocated list
    segments = []
    print("[Benedict] Transcribing audio...")
    
    for i in tqdm(range(total_segments), desc="Transcribing", unit="segment"):
        start = i * samples_per_segment
        end = min(start + samples_per_segment, len(audio))
        
        chunk = whisper.pad_or_trim(audio[start:end])
        mel_chunk = whisper.log_mel_spectrogram(chunk, n_mels=n_mels).to(model_device, dtype=model_dtype)
        
        # Verify shape
        if mel_chunk.shape != (n_mels, 3000):
            print(f"Skipping malformed chunk {i} with shape {mel_chunk.shape}")
            continue
        
        mel_chunk = mel_chunk.unsqueeze(0)
        result = model.decode(mel_chunk, language=detected_lang)
        
        # Extract text
        if hasattr(result, 'text'):
            text = result.text.strip()
        elif isinstance(result, list) and result:
            text = getattr(result[0], 'text', str(result[0])).strip()
        else:
            text = str(result).strip()
        
        if text:
            segments.append(text)
    
    return " ".join(segments), detected_lang


# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benedict: fetch YouTube captions (manual→auto); fall back to Whisper if missing."
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--no-ytsubs", action="store_true",
                        help="Skip YouTube captions and go straight to Whisper")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="Force compute device for Whisper (default: auto)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Whisper batch size for decoding")
    parser.add_argument("--speed", choices=["fast", "balanced", "accurate"], default="fast",
                        help="Whisper decode preset")
    parser.add_argument("--cookies-from-browser",
                        help="yt-dlp: load cookies from a browser")
    parser.add_argument("--cookies", dest="cookies_file",
                        help="yt-dlp: cookie file path")
    
    args = parser.parse_args()
    
    # Try captions first (unless skipped)
    if not args.no_ytsubs:
        # Try youtube-transcript-api
        txt, lang, source = try_fetch_youtube_transcript(args.url, DEFAULT_LANGS)
        if txt:
            out_file = f"transcript_{lang}_yt.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
            print(f"[Benedict] Found YouTube {source} transcript via youtube-transcript-api; saved to {out_file}")
            return
        
        print("[Benedict] youtube-transcript-api returned no transcript.")
        
        # Try yt_dlp captions
        txt2, lang2, source2 = try_fetch_ytdlp_captions(
            args.url,
            preferred_langs=DEFAULT_LANGS,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
        )
        if txt2:
            out_file = f"transcript_{lang2}_yt.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(txt2 + "\n")
            print(f"[Benedict] Found YouTube {source2} captions via yt_dlp; saved to {out_file}")
            return
        
        print("[Benedict] yt_dlp found no captions.")
    
    # Fall back to Whisper
    print("[Benedict] No YouTube captions found, falling back to Whisper...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        print("[Benedict] Downloading audio...")
        audio_path, duration = download_audio(
            args.url, tmpdir,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
        )
        
        model_size = select_model(duration)
        forced = None if args.device == "auto" else args.device
        
        transcript, lang = transcribe_with_whisper(
            audio_path, model_size, forced_device=forced,
        )
        
        out_file = f"transcript_{lang}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"[Benedict] Whisper transcript saved to {out_file}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
