import os
import pandas as pd

from classification.train.tma_instance.paths import get_root_dir

root_dir = get_root_dir()

to_replace_root_dir = 'E:\\'
print(root_dir)
print(to_replace_root_dir)
actual_root_dir = root_dir  # 'D:\\'

base_dir = os.path.join(root_dir, 'DatasetInstancesTMAClassification')

dataset_csv_1 = os.path.join(base_dir, 'dataset_tma_classification_holdout_mask_fill_crop_added.csv')
dataset_csv_2 = os.path.join(base_dir, 'dataset_tma_classification_holdout_mask_fill_merged.csv')

dataset_csv_out_1 = os.path.join(base_dir, 'dataset_tma_classification_holdout_mask_fill_crop_added_out.csv')
dataset_csv_out_2 = os.path.join(base_dir, 'dataset_tma_classification_holdout_mask_fill_merged_out.csv')

for csv_in, csv_out in zip([dataset_csv_1, dataset_csv_2], [dataset_csv_out_1, dataset_csv_out_2]):
    dataset_df = pd.read_csv(csv_in, sep=';')
    print(dataset_df['crop_path'])
    dataset_df['crop_path'] = dataset_df['crop_path'].str.replace(to_replace_root_dir, actual_root_dir, regex=False)
    print(dataset_df['crop_path'])
    dataset_df.to_csv(csv_out, sep=';', index=False)
