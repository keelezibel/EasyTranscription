import os
import shutil
from glob import glob
from dotenv import load_dotenv

load_dotenv()


output_folders = os.path.join(os.getenv("DATA_FOLDER"), os.getenv("OUTPUT_FOLDER"))
all_transcripts = f"{output_folders}{os.path.sep}**{os.path.sep}*.txt"
transcripts = glob(all_transcripts)

transcript_folder = os.path.join(
    os.getenv("DATA_FOLDER"), os.getenv("TRANSCRIPT_FOLDER")
)

print(f"Copying {len(transcripts)} files...")
for f in transcripts:
    dst_file = os.path.join(transcript_folder, os.path.basename(f))
    shutil.copy(f, dst_file)
print(f"Done transferring {len(transcripts)} files.")
