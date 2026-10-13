import os

som_file = '/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation_coco/SOM'
files = os.listdir(som_file)
for f in files:
    new_f = f'0{f}'
    org_dir = os.path.join(som_file, f)
    new_dir = os.path.join(som_file, new_f)
    os.rename(org_dir, new_dir)