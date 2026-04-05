"""
Download a public sports video using yt-dlp.
Usage: python download_video.py --url <youtube_url>
"""
import subprocess
import argparse
import os

def download_video(url: str, output_path: str = "input_video.mp4"):
    """Download video from a public URL using yt-dlp."""
    print(f"[INFO] Downloading video from: {url}")
    command = [
        "yt-dlp",
        "-f", "best[ext=mp4][height<=720]",  # 720p max for speed
        "-o", output_path,
        url
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[SUCCESS] Video saved to: {output_path}")
    else:
        print(f"[ERROR] Download failed:\n{result.stderr}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Public YouTube URL")
    parser.add_argument("--output", type=str, default="input_video.mp4")
    args = parser.parse_args()
    download_video(args.url, args.output)