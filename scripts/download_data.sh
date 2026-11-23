#!/usr/bin/env bash
# Download BloodMNIST (224 px) from the official MedMNIST+ Zenodo record using
# several resumable connections, then verify the MD5 published in medmnist.INFO.
# Useful when Zenodo is slow per connection. Re-running resumes where it stopped.
#
#   bash scripts/download_data.sh            # from the repo root
set -euo pipefail

URL="https://zenodo.org/records/10519652/files/bloodmnist_224.npz?download=1"
MD5="b718ff6835fcbdb22ba9eacccd7b2601"
SIZE=1540731655
PARTS=8
OUT_DIR="data"
OUT="$OUT_DIR/bloodmnist_224.npz"
TMP="$OUT_DIR/.parts"

mkdir -p "$TMP"
if [[ -f "$OUT" ]]; then echo "$OUT already exists"; exit 0; fi

fetch_part() {  # fetch_part <index> <first byte> <last byte>
  local i=$1 start=$2 end=$3 f="$TMP/part$1" have
  for attempt in $(seq 1 200); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    (( start + have > end )) && return 0
    curl -s -L --fail --connect-timeout 30 --speed-limit 1000 --speed-time 60 \
      -r "$((start + have))-$end" "$URL" >> "$f" || sleep 5
  done
  echo "part $i did not finish" >&2; return 1
}

chunk=$(( (SIZE + PARTS - 1) / PARTS ))
pids=()
for ((i = 0; i < PARTS; i++)); do
  start=$(( i * chunk )); end=$(( start + chunk - 1 )); (( end >= SIZE )) && end=$((SIZE - 1))
  fetch_part "$i" "$start" "$end" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for ((i = 0; i < PARTS; i++)); do cat "$TMP/part$i"; done > "$OUT.tmp"
got=$(md5sum "$OUT.tmp" | cut -d' ' -f1)
if [[ "$got" != "$MD5" ]]; then
  echo "MD5 mismatch ($got != $MD5); delete $TMP and retry" >&2; rm -f "$OUT.tmp"; exit 1
fi
mv "$OUT.tmp" "$OUT" && rm -rf "$TMP"
echo "downloaded and verified $OUT"
