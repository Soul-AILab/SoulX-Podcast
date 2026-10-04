#!/bin/bash
set -euo pipefail

cd /content/SoulX-Podcast
export PYTHONPATH="/content/SoulX-Podcast"

PROMPT_TEXT="The stars, the galaxies, and the cosmos are objects of great interest to young people."
PROMPT_AUDIO="hello_copy.wav"
MODEL_PATH="pretrained_models/SoulX-Podcast-1.7B"
SEED=7
MAX_CHARS_PER_CHUNK=800                 # target ~55s per chunk
SCRIPT_FILE="/content/SoulX-Podcast/script.txt"
CHUNK_DIR="/content/SoulX-Podcast/chunks"
WAV_DIR="/content/SoulX-Podcast/outputs/wavs"
FINAL_OUTPUT="/content/SoulX-Podcast/outputs/lda_full.wav"

mkdir -p "${CHUNK_DIR}" "${WAV_DIR}" "$(dirname "${FINAL_OUTPUT}")"

# ------------------------------------------------------------------
# 1.  Split on sentence boundaries (. ! ?) — every chunk ≤ MAX_CHARS
# ------------------------------------------------------------------
echo "=== Splitting script.txt on sentence boundaries ==="
TOTAL_CHARS=$(wc -m < "${SCRIPT_FILE}")
echo "Total characters: ${TOTAL_CHARS}"

python3 -c "
import re
text = open('${SCRIPT_FILE}').read().strip()
sentences = re.split(r'(?<=[.!?])\s+', text)     # keep the punctuation with its sentence
chunks = []
current = ''
for s in sentences:
    s = s.strip()
    if not s or len(s) > ${MAX_CHARS_PER_CHUNK}:
        # A single sentence exceeds the limit — force-split on comma/space
        while len(s) > ${MAX_CHARS_PER_CHUNK}:
            head, s = s[:${MAX_CHARS_PER_CHUNK}], s[${MAX_CHARS_PER_CHUNK}:]
            comma_at = head.rfind(',')
            space_at = head.rfind(' ')
            split_at = comma_at if comma_at > ${MAX_CHARS_PER_CHUNK} * 0.5 else space_at
            if split_at > ${MAX_CHARS_PER_CHUNK} * 0.3:
                s = head[split_at:].lstrip() + ' ' + s
                head = head[:split_at].rstrip()
            if current:
                chunks.append(current.strip())
                current = ''
            chunks.append(head.strip())
            continue

    if not current:
        current = s
    elif len(current) + 1 + len(s) <= ${MAX_CHARS_PER_CHUNK}:
        current = current + ' ' + s
    else:
        chunks.append(current.strip())
        current = s

if current.strip():
    chunks.append(current.strip())

# Write each chunk to its own file
for i, chunk in enumerate(chunks):
    path = '${CHUNK_DIR}/chunk_{:04d}.txt'.format(i)
    with open(path, 'w') as f:
        f.write(chunk)
    print(f'  chunk_{i:04d}.txt  —  {len(chunk)} chars')

print(f'')
print(f'Total chunks: {len(chunks)}')
"

TOTAL_CHUNKS=$(ls -1 "${CHUNK_DIR}"/chunk_*.txt 2>/dev/null | wc -l)
echo ""

# ------------------------------------------------------------------
# 2.  Synthesise every chunk
# ------------------------------------------------------------------
echo "=== Synthesising ${TOTAL_CHUNKS} chunk(s) ==="

for i in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    IDX=$(printf "%04d" $i)
    CHUNK_TXT="${CHUNK_DIR}/chunk_${IDX}.txt"
    WAV_OUT="${WAV_DIR}/chunk_${IDX}.wav"
    CHARS=$(wc -m < "${CHUNK_TXT}")

    echo "  [$(($i + 1))/${TOTAL_CHUNKS}] ${IDX}.txt → ${IDX}.wav  (${CHARS} chars)"

    python cli/tts.py \
        --prompt_text "${PROMPT_TEXT}" \
        --prompt_audio "${PROMPT_AUDIO}" \
        --text "${CHUNK_TXT}" \
        --model_path "${MODEL_PATH}" \
        --output_path "${WAV_OUT}" \
        --seed ${SEED}
done

echo ""
echo "=== All chunks synthesised ==="

# ------------------------------------------------------------------
# 3.  Concatenate all clips into one WAV
# ------------------------------------------------------------------
echo "=== Concatenating with ffmpeg ==="
CONCAT_LIST="${CHUNK_DIR}/concat_list.txt"
> "${CONCAT_LIST}"

for wav in "${WAV_DIR}"/chunk_*.wav; do
    echo "file '$(realpath "${wav}")'" >> "${CONCAT_LIST}"
done

ffmpeg -y -f concat -safe 0 -i "${CONCAT_LIST}" -c copy "${FINAL_OUTPUT}"

echo ""
echo "=== Done ==="
echo "Chunks:        ${WAV_DIR}/"
echo "Concatenated:  ${FINAL_OUTPUT}"
