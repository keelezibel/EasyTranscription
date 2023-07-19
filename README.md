# EasyTranscription
Easy wrapper for Whisper with Voice Activity Detection and Source Separation.

Currently, source separation only supports up to 10mins of audio but we can still transcribe audio of any length. 

## Technologies Used
1. Whisper
2. whisper.cpp
3. Spleeter
4. SileroVAD

## Transcription Support
### Step 1: Gather list of URLS and save as `data/links.txt`

### Step 2: Download [models](https://drive.google.com/file/d/1iw9B97j9BdQ7ms3yMJitM37V3iDFAncj/view?usp=sharing) and replace with `models` folder
Do note that only following models are included. If you want other model sizes, download directly from the corresponding sites.
For whisper **(large, medium, small and tiny)** and whisper.cpp **(medium, small and tiny)**

Change the environment variables `WHISPER_MODEL_SIZE` or `WHISPERCPP_MODEL_SIZE` for whisper or whisper.cpp accordingly.

### Step 3: Docker Compose up
```
# GPU (check that you have CUDA and cuDNN installed)
docker-compose -f docker-compose-gpu.yaml up -d
# CPU only
docker-compose -f docker-compose-cpu.yaml up -d
```

### Step 4: Crawl all videos
```python
python3 src/video_scrapper.py
```

### Step 5: Transcribe
```python
python3 src/transcribe.py
```

## VTT support (standalone script)
### Gather VTT Formats transcripts from YouTube
`yt-dlp --write-auto-sub --sub-format vtt --skip-download <YouTube Link>`
### Convert VTT transcripts to TXT only
`find . -name "*.vtt" -exec python3 src/vtt2text.py  {} \;`