#!/usr/bin/env bash
# Monitor a HuggingFace model download.
#
# Usage:  tools/watch_hf_download.sh <repo-id> [expected-GiB]
#   e.g.  tools/watch_hf_download.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers 32
#
# `hf download` reports an aggregate byte counter that includes partially
# written *.incomplete staging files, so the figure it prints can sit near the
# final total while very little has actually been committed. This shows both:
# bytes on disk, and how many shards are genuinely finished.
set -u
REPO="${1:?usage: $0 <repo-id> [expected-GiB]}"
EXPECT_GIB="${2:-0}"
DIR="$HOME/.cache/huggingface/hub/models--${REPO//\//--}"

prev=0; prev_t=0
while true; do
    [ -d "$DIR" ] || { echo "waiting for $DIR …"; sleep 5; continue; }
    now=$(du -sb "$DIR" 2>/dev/null | cut -f1); now=${now:-0}
    t=$(date +%s)
    done_n=$(find "$DIR/blobs" -type f ! -name '*.incomplete' 2>/dev/null | wc -l)
    part_n=$(find "$DIR" -name '*.incomplete' 2>/dev/null | wc -l)

    rate=0
    if [ "$prev" -gt 0 ] && [ "$t" -gt "$prev_t" ]; then
        rate=$(( (now - prev) / (t - prev_t) / 1048576 ))
    fi

    eta="--"
    if [ "$EXPECT_GIB" -gt 0 ] && [ "$rate" -gt 0 ]; then
        left=$(( EXPECT_GIB * 1024 - now / 1048576 ))
        [ "$left" -gt 0 ] && eta="$(( left / rate / 60 ))m"
    fi

    printf "\r\033[K%6.2f GiB | %3d MB/s | ETA %-5s | committed %-3s shards | in flight %-2s" \
        "$(awk -v n="$now" 'BEGIN{printf "%.2f", n/1073741824}')" \
        "$rate" "$eta" "$done_n" "$part_n"

    if ! pgrep -f "hf download.*${REPO##*/}" >/dev/null 2>&1; then
        printf "\n"
        if [ "$part_n" -eq 0 ]; then echo "download process finished — no partial shards left"
        else echo "download process gone but $part_n partial shard(s) remain — it was interrupted"; fi
        exit 0
    fi
    prev=$now; prev_t=$t
    sleep 5
done
