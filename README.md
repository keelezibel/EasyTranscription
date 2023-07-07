# EasyTranscription

## Step 1: Gather list of URLS and save as `data/links.txt`

## Step 2: Download [models](https://drive.google.com/file/d/1iw9B97j9BdQ7ms3yMJitM37V3iDFAncj/view?usp=sharing) and replace with `models` folder

## Step 3: Docker Compose up
`docker-compose up -d`

## Step 4: Crawl all videos
```python
python3 src/video_scrapper.py
```

## Step 5: Transcribe
```python
python3 src/transcribe.py
```

## Alternatively, gather VTT Formats transcripts from YouTube
`yt-dlp --write-auto-sub --sub-format vtt --skip-download <YouTube Link>`
## Convert VTT transcripts to TXT only
`find . -name "*.vtt" -exec python3 src/vtt2text.py  {} \;`