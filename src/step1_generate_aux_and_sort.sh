#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0

PYTHON_FILE="step1_generate_aux_and_sort.py"

output_file="/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object3/missing_144.json"

python ${PYTHON_FILE} \
 --id 1 --prompt "a shoe without a string"
 --biased_semantic "shoe string" \
 --bias_absent_semantic "A pair of modern laceless slip-on sneakers, minimalist athletic design, clean all-white color scheme, seamless knit upper with soft breathable mesh texture, sleek lightweight sole with subtle curved contours, no shoelaces, contemporary sports footwear aesthetic, positioned at a slight angle in a studio setup, ultra realistic DSLR product photography, highly detailed materials and stitching, soft diffused lighting, clean white background, photorealistic, sharp focus, commercial footwear catalog style, premium product shot, high resolution" \
 --aux_num 20 --batch_generate 20 --output_file "${output_file}" --task object