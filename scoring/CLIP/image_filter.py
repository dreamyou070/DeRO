import os

base_folder = '/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation_coco'
folder1 = os.path.join(base_folder, 'FLUX')
folder2 = os.path.join(base_folder, 'FLUX_SVD')
folder3 = os.path.join(base_folder, 'FLUX_OPTIMIZE')

os.makedirs(folder2, exist_ok=True)
os.makedirs(folder3, exist_ok=True)

for file in os.listdir(folder1):
    src = os.path.join(folder1, file)

    if file.endswith('_optimize.png'):
        unique = file[:-len('_optimize.png')]
        dst = os.path.join(folder3, f'{unique}.png')

    elif file.endswith('_svd.png'):
        unique = file[:-len('_svd.png')]
        dst = os.path.join(folder2, f'{unique}.png')

    else:
        continue

    os.rename(src, dst)
