# Negation
# /MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation 여기 아래 4개 모델 폴더 있어
# 각 폴더 아래 이미지 있고 파일명은 6자리 id.png 야
# excel = "/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/relation.xlsx"  에서 id 하고 프롬프트 열을 보고
# id 열의 값을 str 로 해서 6자리화해
# 그 이미지하고 해당 프롬프트를 이ㅛㅇ해서
# 각 모델의 CLIP 점수 구하고 시어

# python clipscore.py example_shoe/good_captions.json example_shoe/images/

#python clipscore_p.py \
#  --root_dir /MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation_coco \
#  --excel /MIR/nas1/dreamyou070/NegatingDiffusion/data_final/1_negation_coco.xlsx \
#  --id_col id --prompt_col "positive prompt" \
#  --save_json /MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation_coco/_1_clip/1_negation_coco_clipscore_positive.json

python clipscore_p.py \
  --root_dir /MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/2_negation_art \
  --excel /MIR/nas1/dreamyou070/NegatingDiffusion/data_final/2_negation_vangogh.xlsx \
  --id_col id --prompt_col "positive prompt" \
  --save_json /MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/2_negation_art/_1_clip/1_negation_coco_clipscore_positive.json