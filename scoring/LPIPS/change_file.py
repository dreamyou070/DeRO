import os
from PIL import Image

org_folder = '/MIR/nas1/dreamyou070/NegatingDiffusion/2_concept_negate/3_ours/picasso'
new_folder = "/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/FINAL/2_ART/negative"

datas = os.listdir(org_folder)
for d in datas:
    org_path = os.path.join(org_folder, d)
    new_path = os.path.join(new_folder, d)
    Image.open(new_path).save(org_path)

org_folder = '/MIR/nas1/dreamyou070/NegatingDiffusion/2_concept_negate/3_ours/vangogh'
new_folder = "/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/FINAL/2_ART/negative"

datas = os.listdir(org_folder)
for d in datas:
    org_path = os.path.join(org_folder, d)
    new_path = os.path.join(new_folder, d)
    Image.open(new_path).save(org_path)