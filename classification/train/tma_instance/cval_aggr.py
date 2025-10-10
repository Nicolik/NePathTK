import os
import sys
import cv2
import pandas as pd

base_abs_path = os.path.abspath('.')
print(f"Absolute path: {base_abs_path}")
sys.path.append(base_abs_path)

from classification.train.tma_instance.paths import get_logs_path
from definitions import ROOT_DIR
from semseg.figure_util import hconcat_with_border, vconcat_with_border
from classification.train.tma_instance.params import targets

use_mimickers = False
train_version = "V1"
root_path = ROOT_DIR
folds = 1
logs_path_base = get_logs_path(root_path)

percent_multiplier = 1
ndigits = 3

models = ["resnet34", "densenet121", "efficientnetv2-s", "convnext_small", "swin_v2_b"]
compartments = ('Glomerulus', ('Artery', 'Arteriole'))
compartments_names = ('Glomerulus', 'Artery+Arteriole')

ROC_X = ROC_Y = PR_X = PR_Y = 1000
TSNE_X = UMAP_X = 4000
TSNE_Y = UMAP_Y = 3000
CM_X = CM_Y = 600
# BORDER_COLOR = (0, 0, 0)
# BORDER_COLOR = (100, 100, 100)
BORDER_COLOR = (255, 255, 255)
BORDER_THICKNESS = 0.005  # 0.05

embedding_plot_type = ""

all_summaries_comp = []
all_roc_list_comp = []
all_roc_all_list_comp = []
all_pr_list_comp = []
all_pr_all_list_comp = []

for compartment, comp_name in zip(compartments, compartments_names):
    all_summaries = []
    all_cm_list = []
    all_roc_list = []
    all_roc_all_list = []
    all_pr_list = []
    all_pr_all_list = []
    all_tsne_list = []
    all_umap_list = []

    print(f"Compartment: {compartment}")
    logs_dir = os.path.join(logs_path_base, str(compartment))

    for model in models:
        model_dir = os.path.join(logs_dir, model)
        print(f"Model: {model}, dir: {model_dir}")
        for target in targets:
            aggr_metrics_csv = os.path.join(model_dir, f"cross_validation_{target}_metrics.csv")

            aggr_roc_png = os.path.join(model_dir, f"cross_validation_{target}_ROC.png")
            aggr_roc_all_png = os.path.join(model_dir, f"cross_validation_{target}_ROC_All_Subsets.png")
            aggr_pr_png = os.path.join(model_dir, f"cross_validation_{target}_PR.png")
            aggr_pr_all_png = os.path.join(model_dir, f"cross_validation_{target}_PR_All_Subsets.png")
            aggr_tsne_png = os.path.join(model_dir, f"cross_validation_{target}_TSNE.png")
            aggr_umap_png = os.path.join(model_dir, f"cross_validation_{target}_UMAP.png")
            aggr_cm_png = os.path.join(model_dir, f"cross_validation_{target}_CM.png")

            all_metrics = []

            model_target_cm_list = []
            model_target_roc_list = []
            model_target_roc_all_list = []
            model_target_pr_list = []
            model_target_pr_all_list = []
            model_target_tsne_list = []
            model_target_umap_list = []

            for fold in [f"fold-{i}" for i in range(folds)]:
                fold_dir = os.path.join(model_dir, fold)
                print(f"Fold: {fold}, dir: {fold_dir}")

                roc_all_path = os.path.join(fold_dir, f'ROC_{model}_{target}_{train_version}_All_Subsets.png')
                roc_all_image = cv2.imread(roc_all_path)
                model_target_roc_all_list.append(roc_all_image)

                pr_all_path = os.path.join(fold_dir, f'PR_{model}_{target}_{train_version}_All_Subsets.png')
                pr_all_image = cv2.imread(pr_all_path)
                model_target_pr_all_list.append(pr_all_image)

                for subset in ['Val', 'Test']:

                    filename_metric_path = os.path.join(fold_dir, f'Metrics_{model}_{target}_{train_version}_{subset}.csv')
                    print(f"Attempting to open {filename_metric_path}")
                    metric_df = pd.read_csv(filename_metric_path, sep=';')
                    metric_df.insert(0, "subset", subset)
                    all_metrics.append(metric_df)
                    print(metric_df.head())

                    name_embedding_file = f'{model}_trained_{target}_{train_version}_{subset}'

                    roc_path = os.path.join(fold_dir, f'ROC_{model}_{target}_{train_version}_{subset}.png')
                    pr_path = os.path.join(fold_dir, f'PR_{model}_{target}_{train_version}_{subset}.png')
                    cm_path = os.path.join(fold_dir, f'CM_{model}_{target}_{train_version}_{subset}.png')
                    itsne_path = os.path.join(fold_dir, f'{embedding_plot_type}tsne_{name_embedding_file}.png')
                    iumap_path = os.path.join(fold_dir, f'{embedding_plot_type}umap_{name_embedding_file}.png')

                    roc_image = cv2.imread(roc_path)
                    pr_image = cv2.imread(pr_path)
                    cm_image = cv2.imread(cm_path)
                    tsne_image = cv2.imread(itsne_path)
                    umap_image = cv2.imread(iumap_path)
                    print(f"[Shapes] ROC: {roc_image.shape}, PR: {pr_image.shape}, CM: {cm_image.shape}, "
                          f"TSNE: {tsne_image.shape}, UMAP: {umap_image.shape}")
                    model_target_cm_list.append(cm_image)
                    model_target_roc_list.append(roc_image)
                    model_target_pr_list.append(pr_image)
                    model_target_tsne_list.append(tsne_image)
                    model_target_umap_list.append(umap_image)

            model_target_cm = vconcat_with_border(model_target_cm_list, int(CM_X * BORDER_THICKNESS), BORDER_COLOR)
            model_target_roc = vconcat_with_border(model_target_roc_list, int(ROC_X * BORDER_THICKNESS), BORDER_COLOR)
            model_target_roc_all = vconcat_with_border(model_target_roc_all_list, int(ROC_X * BORDER_THICKNESS), BORDER_COLOR)
            model_target_pr = vconcat_with_border(model_target_pr_list, int(PR_X * BORDER_THICKNESS), BORDER_COLOR)
            model_target_pr_all = vconcat_with_border(model_target_pr_all_list, int(PR_X * BORDER_THICKNESS), BORDER_COLOR)
            model_target_tsne = vconcat_with_border(model_target_tsne_list, int(TSNE_X * BORDER_THICKNESS), BORDER_COLOR)
            model_target_umap = vconcat_with_border(model_target_umap_list, int(UMAP_X * BORDER_THICKNESS), BORDER_COLOR)

            all_cm_list.append(model_target_cm)
            all_roc_list.append(model_target_roc)
            all_roc_all_list.append(model_target_roc_all)
            all_pr_list.append(model_target_pr)
            all_pr_all_list.append(model_target_pr_all)
            all_tsne_list.append(model_target_tsne)
            all_umap_list.append(model_target_umap)

            cv2.imwrite(aggr_roc_png, model_target_roc)
            cv2.imwrite(aggr_roc_all_png, model_target_roc_all)
            cv2.imwrite(aggr_pr_png, model_target_pr)
            cv2.imwrite(aggr_pr_all_png, model_target_pr_all)
            cv2.imwrite(aggr_cm_png, model_target_cm)
            cv2.imwrite(aggr_tsne_png, model_target_tsne)
            cv2.imwrite(aggr_umap_png, model_target_umap)

            combined_df = pd.concat(all_metrics)

            formatted_df = combined_df.copy()
            cols_to_format = ['accuracy', 'precision', 'recall', 'specificity', 'auroc', 'auprc']

            # Apply formatting only to these columns
            for col in cols_to_format:
                if col in formatted_df.columns:  # safety check
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: "--" if pd.isna(x) else f"{round(x * percent_multiplier, ndigits):.{ndigits}f}"
                    )

            summary_df = formatted_df.reset_index()
            summary_df.to_csv(aggr_metrics_csv, index=False, sep=';')
            # summary_df.insert(0, "target", target)
            summary_df.insert(0, "compartment", comp_name)
            summary_df.insert(1, "model", model)
            summary_df = summary_df.drop(columns=['index'])
            all_summaries.append(summary_df)
            print(f"Saved aggregated metrics to: {aggr_metrics_csv}")

    full_df = pd.concat(all_summaries, ignore_index=True)
    full_output_csv = os.path.join(logs_dir, "all_models_crossval_metrics.csv")
    full_df.to_csv(full_output_csv, index=False, sep=';')
    print(f"Saved all models' aggregated metrics to: {full_output_csv}")

    keys = ['compartment', 'model', 'subset']
    overall_acc = (
        full_df.loc[full_df['class'] == 'OVERALL', keys + ['accuracy']]
        .drop_duplicates(subset=keys, keep='last')  # in case of duplicates
        .set_index(keys)['accuracy']
    )
    tma = full_df.loc[full_df['class'] == 'TMA'].copy()
    tma['accuracy'] = tma[keys].apply(lambda r: overall_acc.get(tuple(r)), axis=1)
    tma_df = tma.drop(columns=['class'])
    tma_output_csv = os.path.join(logs_dir, "all_models_crossval_metrics_tma.csv")
    tma_df.to_csv(tma_output_csv, index=False, sep=';')

    all_cm = hconcat_with_border(all_cm_list, int(CM_Y * BORDER_THICKNESS), BORDER_COLOR)
    all_roc_all = hconcat_with_border(all_roc_all_list, int(ROC_Y * BORDER_THICKNESS), BORDER_COLOR)
    all_roc = hconcat_with_border(all_roc_list, int(ROC_Y * BORDER_THICKNESS), BORDER_COLOR)
    all_pr = hconcat_with_border(all_pr_list, int(PR_Y * BORDER_THICKNESS), BORDER_COLOR)
    all_pr_all = hconcat_with_border(all_pr_all_list, int(PR_Y * BORDER_THICKNESS), BORDER_COLOR)
    all_tsne = hconcat_with_border(all_tsne_list, int(TSNE_Y * BORDER_THICKNESS), BORDER_COLOR)
    all_umap = hconcat_with_border(all_umap_list, int(UMAP_Y * BORDER_THICKNESS), BORDER_COLOR)

    all_summaries_comp.append(tma_df)
    all_roc_list_comp.append(all_roc)
    all_roc_all_list_comp.append(all_roc_all)
    all_pr_list_comp.append(all_pr)
    all_pr_all_list_comp.append(all_pr_all)

    all_roc_png = os.path.join(logs_dir, "all_models_crossval_ROC.png")
    all_roc_all_png = os.path.join(logs_dir, "all_models_crossval_ROC_All_Subsets.png")
    all_pr_png = os.path.join(logs_dir, "all_models_crossval_PR.png")
    all_pr_all_png = os.path.join(logs_dir, "all_models_crossval_PR_All_Subsets.png")
    all_cm_png = os.path.join(logs_dir, "all_models_crossval_CM.png")
    all_tsne_png = os.path.join(logs_dir, "all_models_crossval_TSNE.png")
    all_umap_png = os.path.join(logs_dir, "all_models_crossval_UMAP.png")

    cv2.imwrite(all_cm_png, all_cm)
    cv2.imwrite(all_roc_png, all_roc)
    cv2.imwrite(all_roc_all_png, all_roc_all)
    cv2.imwrite(all_pr_png, all_pr)
    cv2.imwrite(all_pr_all_png, all_pr_all)
    cv2.imwrite(all_tsne_png, all_tsne)
    cv2.imwrite(all_umap_png, all_umap)

full_df = pd.concat(all_summaries_comp, ignore_index=True)
full_output_csv = os.path.join(logs_path_base, f"all_models_metrics_Mimickers_{use_mimickers}.csv")
full_df.to_csv(full_output_csv, index=False, sep=';')
print(f"Saved all models' aggregated metrics to: {full_output_csv}")

all_roc_comp = vconcat_with_border(all_roc_list_comp, int(ROC_Y * BORDER_THICKNESS), BORDER_COLOR)
all_roc_png = os.path.join(logs_path_base, "all_models_ROC.png")
cv2.imwrite(all_roc_png, all_roc_comp)

all_roc_comp_all = vconcat_with_border(all_roc_all_list_comp, int(ROC_Y * BORDER_THICKNESS), BORDER_COLOR)
all_roc_all_png = os.path.join(logs_path_base, f"all_models_ROC_All_Subsets_Mimickers_{use_mimickers}.png")
cv2.imwrite(all_roc_all_png, all_roc_comp_all)

all_pr_comp = vconcat_with_border(all_pr_list_comp, int(PR_Y * BORDER_THICKNESS), BORDER_COLOR)
all_pr_png = os.path.join(logs_path_base, "all_models_PR.png")
cv2.imwrite(all_pr_png, all_pr_comp)

all_pr_comp_all = vconcat_with_border(all_pr_all_list_comp, int(PR_Y * BORDER_THICKNESS), BORDER_COLOR)
all_pr_all_png = os.path.join(logs_path_base, f"all_models_PR_All_Subsets_Mimickers_{use_mimickers}.png")
cv2.imwrite(all_pr_all_png, all_pr_comp_all)