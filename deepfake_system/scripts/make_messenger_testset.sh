#!/usr/bin/env bash
# Build a genuinely re-encoded copy of your test videos.
#
# The simulator in data/degrade.py is a fast approximation used for
# training. For the numbers that go in your report, you want real H.264
# transcodes. These presets follow what WhatsApp, Telegram and Instagram
# actually do to an uploaded video.
#
#   ./make_messenger_testset.sh /path/to/test_videos /path/to/output
#
# Best of all: send ten real videos through WhatsApp yourself, download
# them back, and put them in a folder called ground_truth_wa/. Two hours
# of work, and it is the single most defensible slide in your viva.

set -euo pipefail
SRC="${1:?usage: make_messenger_testset.sh SRC_DIR OUT_DIR}"
OUT="${2:?usage: make_messenger_testset.sh SRC_DIR OUT_DIR}"

mkdir -p "$OUT"/{whatsapp,whatsapp_forwarded,telegram,instagram}

shopt -s nullglob
for f in "$SRC"/*.{mp4,avi,mov,mkv,webm}; do
  name="$(basename "${f%.*}")"

  # WhatsApp: H.264 baseline, 480p cap, ~1.1 Mbps, 4:2:0, metadata stripped.
  ffmpeg -y -loglevel error -i "$f" \
    -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p \
    -vf "scale='min(848,iw)':'min(480,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -b:v 1100k -maxrate 1300k -bufsize 2000k -r 30 \
    -c:a aac -b:a 64k -ar 44100 -map_metadata -1 -movflags +faststart \
    "$OUT/whatsapp/${name}_wa.mp4"

  # Forwarded twice — each hop re-encodes, and this is the realistic case.
  ffmpeg -y -loglevel error -i "$OUT/whatsapp/${name}_wa.mp4" \
    -c:v libx264 -profile:v baseline -pix_fmt yuv420p \
    -vf "scale='min(640,iw)':'min(360,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -b:v 700k -maxrate 850k -bufsize 1400k \
    -c:a aac -b:a 48k -map_metadata -1 \
    "$OUT/whatsapp_forwarded/${name}_wa2.mp4"

  # Telegram: kinder, higher bitrate, 720p.
  ffmpeg -y -loglevel error -i "$f" \
    -c:v libx264 -profile:v main -pix_fmt yuv420p \
    -vf "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -crf 26 -c:a aac -b:a 96k -map_metadata -1 \
    "$OUT/telegram/${name}_tg.mp4"

  # Instagram / TikTok style: vertical crop, aggressive rate control.
  ffmpeg -y -loglevel error -i "$f" \
    -c:v libx264 -profile:v main -pix_fmt yuv420p \
    -vf "scale=720:-2,crop=720:min(1280\,ih)" \
    -b:v 1500k -maxrate 1800k -bufsize 3000k -r 30 \
    -c:a aac -b:a 96k -map_metadata -1 \
    "$OUT/instagram/${name}_ig.mp4"

  echo "done: $name"
done

echo
echo "Now score each folder separately and put the four numbers in your"
echo "results table. The gap between 'clean' and 'whatsapp_forwarded' is"
echo "the finding, not something to hide."
