# Benedict

> *The script is named **Benedict** in honor of the Benedictine monks who historically dedicated their lives to meticulously transcribing and preserving classical texts.*

Benedict is a Python script that provides a comprehensive, sequential fallback approach to obtaining transcriptions for YouTube videos. It attempts to fetch existing captions using the most efficient methods first and seamlessly falls back to transcribing the video's audio using an AI model (OpenAI's Whisper) if no captions are available.

## How it Works

When you provide a YouTube URL, Benedict processes the request in three stages:

1. **YouTube API (Fastest):** It first tries to grab manually-created or auto-generated captions directly using the `youtube-transcript-api`.
2. **yt-dlp Subtitles (Fallback):** If the API fails, it falls back to `yt-dlp` to search for and extract video subtitles.
3. **OpenAI Whisper (AI Transcription):** If no captions exist anywhere (or if you intentionally skip checking for them), Benedict downloads the audio locally and runs it through the Whisper AI model to generate a brand fresh transcript. 
    - It automatically selects the optimal Whisper model size (`base`, `small`, `medium`, `large`) based on the length of the video.
    - It detects the spoken language by sampling a chunk from the middle of the audio.
    - It automatically utilizes hardware acceleration (like NVIDIA CUDA or Apple Silicon MPS) if available.

## Environment Setup

To ensure all necessary dependencies are correctly set up without conflicting with your global Python packages, please create a virtual environment using the provided `requirements.txt` file.

1. **Navigate to the project directory:**
   ```bash
   cd path/to/project-directory
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```
3. **Activate the environment:**
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **Windows (Command Prompt):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python benedict.py <YouTube_URL> [options]
```

### Positional Arguments
- `url`: The URL of the YouTube video you want to transcribe.

### Options / Flags

- `--no-ytsubs`
  - **What it does:** Completely skips checking YouTube for existing captions and immediately downloads the audio to run inference with the Whisper model.
  - **When to use:** When you specifically want a fresh, potentially more accurate Whisper transcription instead of relying on YouTube's auto-generated captions.

- `--device {auto, cpu, mps, cuda}`
  - **What it does:** Forces Whisper to run on a specific compute device.
  - **Default:** `auto` (Benedict auto-detects if you have CUDA or MPS available. If not, it defaults to CPU).
  - **When to use:** If auto-detection gets confused, or if you want to intentionally force the script to use the CPU instead of utilizing your GPU.

- `--batch-size <integer>`
  - **What it does:** Modifies the batch size used for Whisper decoding.
  - **Default:** `8`

- `--speed {fast, balanced, accurate}`
  - **What it does:** Modifies the decoding preset for Whisper, adjusting the tradeoff between transcription speed and accuracy. 
  - **Default:** `fast`

- `--cookies-from-browser <browser_name>`
  - **What it does:** Instructs `yt-dlp` to extract your YouTube session cookies from a specified browser (e.g., `chrome`, `firefox`, `safari`, `edge`).
  - **When to use:** Required when you need to transcribe age-restricted videos, private videos, or members-only videos that mandate being logged into a YouTube account.

- `--cookies <file_path>`
  - **What it does:** Allows you to pass a traditional `.txt` cookie file to `yt-dlp`. 
  - **When to use:** An alternative to `--cookies-from-browser`, commonly used when running this script headless on a server where a browser isn't installed.
