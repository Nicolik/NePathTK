import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import argparse
import time
import cv2
import shutil
import numpy as np
from semseg.tma.util import area_filter, detect_and_solve_touching_objects
from semseg.qupath.utils import (COLOR_GLOMERULUS_TUPLE, COLOR_ARTERY_TUPLE, COLOR_ARTERIOLE_TUPLE,
                                 COLOR_CORTEX_TUPLE, COLOR_MEDULLA_TUPLE, COLOR_CAPSULE_OTHER_TUPLE,
                                 COLOR_BIOPSY_TISSUE_TUPLE, COLOR_BACKGROUND_TUPLE, COLOR_IFTA_TUPLE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wsi', type=str, help='wsi name without extension', required=True)
    parser.add_argument('--cmc-dir', type=str, help="Directory with the output for Cortex/Medulla/CapsuleOther", required=True)
    parser.add_argument('--aag-dir', type=str, help="Directory with the output for Glomerulus/Artery/Arteriole", required=True)
    parser.add_argument('--ifta-dir', type=str, help="Directory with the output for IFTACortex", required=False)
    parser.add_argument('--pp-dir', type=str, help="Directory where to store post-processing results", required=True)
    parser.add_argument('--observ-dir', type=str, help="Directory where to keep original WSI masks", required=True)
    parser.add_argument('--color-dir', type=str, help="Directory where to export colored masks", required=True)

    args = parser.parse_args()

    wsi_name = args.wsi
    cortex_medulla_capsule_mask_dir = args.cmc_dir
    glomerulus_artery_arteriole_mask_dir = args.aag_dir
    ifta_cortex_mask_dir = args.ifta_dir
    post_processed_mask_dir = args.pp_dir
    observ_mask_dir = args.observ_dir
    color_dir = args.color_dir

    os.makedirs(post_processed_mask_dir, exist_ok=True)
    os.makedirs(observ_mask_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    cortex_wsi_mask = f"{wsi_name}_Cortex_mask.png"
    medulla_wsi_mask = f"{wsi_name}_Medulla_mask.png"
    capsule_wsi_mask = f"{wsi_name}_CapsuleOther_mask.png"
    glomerulus_wsi_mask = f"{wsi_name}_Glomerulus_mask.png"
    arteriole_wsi_mask = f"{wsi_name}_Arteriole_mask.png"
    artery_wsi_mask = f"{wsi_name}_Artery_mask.png"
    ifta_wsi_mask = f"{wsi_name}_IFTACortex_mask.png"

    # Input masks paths
    cortex_mask_path = os.path.join(cortex_medulla_capsule_mask_dir, cortex_wsi_mask)
    medulla_mask_path = os.path.join(cortex_medulla_capsule_mask_dir, medulla_wsi_mask)
    capsule_mask_path = os.path.join(cortex_medulla_capsule_mask_dir, capsule_wsi_mask)
    glomerulus_mask_path = os.path.join(glomerulus_artery_arteriole_mask_dir, glomerulus_wsi_mask)
    arteriole_mask_path = os.path.join(glomerulus_artery_arteriole_mask_dir, arteriole_wsi_mask)
    artery_mask_path = os.path.join(glomerulus_artery_arteriole_mask_dir, artery_wsi_mask)

    # Observ masks paths [development/debug only]
    cortex_mask_observ_path = os.path.join(observ_mask_dir, cortex_wsi_mask)
    medulla_mask_observ_path = os.path.join(observ_mask_dir, medulla_wsi_mask)
    capsule_mask_observ_path = os.path.join(observ_mask_dir, capsule_wsi_mask)
    glomerulus_mask_observ_path = os.path.join(observ_mask_dir, glomerulus_wsi_mask)
    arteriole_mask_observ_path = os.path.join(observ_mask_dir, arteriole_wsi_mask)
    artery_mask_observ_path = os.path.join(observ_mask_dir, artery_wsi_mask)


    # Copy masks
    shutil.copyfile(cortex_mask_path, cortex_mask_observ_path)
    shutil.copyfile(medulla_mask_path, medulla_mask_observ_path)
    shutil.copyfile(capsule_mask_path, capsule_mask_observ_path)
    shutil.copyfile(glomerulus_mask_path, glomerulus_mask_observ_path)
    shutil.copyfile(arteriole_mask_path, arteriole_mask_observ_path)
    shutil.copyfile(artery_mask_path, artery_mask_observ_path)

    if ifta_cortex_mask_dir:
        ifta_mask_path = os.path.join(ifta_cortex_mask_dir, ifta_wsi_mask)
        ifta_mask_observ_path = os.path.join(observ_mask_dir, ifta_wsi_mask)
        shutil.copyfile(ifta_mask_path, ifta_mask_observ_path)
        ifta_mask = cv2.imread(ifta_mask_path, cv2.IMREAD_GRAYSCALE)

    # Output masks paths
    cortex_mask_output_path = os.path.join(post_processed_mask_dir, cortex_wsi_mask)
    medulla_mask_output_path = os.path.join(post_processed_mask_dir, medulla_wsi_mask)
    capsule_mask_output_path = os.path.join(post_processed_mask_dir, capsule_wsi_mask)
    glomerulus_mask_output_path = os.path.join(post_processed_mask_dir, glomerulus_wsi_mask)
    arteriole_mask_output_path = os.path.join(post_processed_mask_dir, arteriole_wsi_mask)
    artery_mask_output_path = os.path.join(post_processed_mask_dir, artery_wsi_mask)
    ifta_mask_output_path = os.path.join(post_processed_mask_dir, ifta_wsi_mask)

    # Cortex Medulla Capsule
    cortex_mask = cv2.imread(cortex_mask_path, cv2.IMREAD_GRAYSCALE)
    medulla_mask = cv2.imread(medulla_mask_path, cv2.IMREAD_GRAYSCALE)
    capsule_mask = cv2.imread(capsule_mask_path, cv2.IMREAD_GRAYSCALE)
    cortex_medulla_capsule_mask = cv2.bitwise_or(cv2.bitwise_or(cortex_mask, medulla_mask), capsule_mask)
    cmc_Y, cmc_X = cortex_mask.shape

    # Glomerulus Artery Arteriole
    glomerulus_mask = cv2.imread(glomerulus_mask_path, cv2.IMREAD_GRAYSCALE)
    arteriole_mask = cv2.imread(arteriole_mask_path, cv2.IMREAD_GRAYSCALE)
    artery_mask = cv2.imread(artery_mask_path, cv2.IMREAD_GRAYSCALE)
    aag_Y, aag_X = glomerulus_mask.shape

    # IFTACortex
    if ifta_cortex_mask_dir is None:
        ifta_mask = np.zeros_like(glomerulus_mask)
    ifta_Y, ifta_X = ifta_mask.shape

    print(f"Image: {wsi_name}")
    print(f"Bioptic Tissue              Shape: {cortex_medulla_capsule_mask.shape}")
    print(f"Cortex/Medulla/Capsule      Shape: {cortex_mask.shape}, {medulla_mask.shape}, {capsule_mask.shape}")
    print(f"Glomerulus/Artery/Arteriole Shape: {glomerulus_mask.shape}, {artery_mask.shape}, {arteriole_mask.shape}")
    print(f"IFTACortex                  Shape: {ifta_mask.shape}")

    # Starting post-processing
    start_pp_time = time.time()

    # Glomerulus/Artery/Arteriole post-processing
    upscaled_bioptic_tissue_mask = cv2.resize(cortex_medulla_capsule_mask, (aag_X, aag_Y), interpolation=cv2.INTER_NEAREST_EXACT)
    upscaled_cortex_medulla_capsule_mask = cv2.resize(cortex_medulla_capsule_mask, (aag_X, aag_Y), interpolation=cv2.INTER_NEAREST_EXACT)
    upscaled_cortex_mask = cv2.resize(cortex_mask, (aag_X, aag_Y), interpolation=cv2.INTER_NEAREST_EXACT)

    # Removole Glomerulus outside the tissue boundaries
    # glomerulus_mask[upscaled_bioptic_tissue_mask == 0] = 0
    # glomerulus_mask[upscaled_cortex_medulla_capsule_mask == 0] = 0

    # Remove Artery, Arteriole, outside the tissue boundaries
    artery_mask[upscaled_bioptic_tissue_mask == 0] = 0
    artery_mask[upscaled_cortex_medulla_capsule_mask == 0] = 0
    arteriole_mask[upscaled_bioptic_tissue_mask == 0] = 0
    arteriole_mask[upscaled_cortex_medulla_capsule_mask == 0] = 0
    ifta_mask[upscaled_bioptic_tissue_mask == 0] = 0
    # ifta_mask[upscaled_cortex_medulla_capsule_mask == 0] = 0
    ifta_mask[upscaled_cortex_mask == 0] = 0

    # Glomerulus > Artery > Arteriole > IFTACortex
    artery_mask[glomerulus_mask != 0] = 0
    arteriole_mask[glomerulus_mask != 0] = 0
    arteriole_mask[artery_mask != 0] = 0
    ifta_mask[glomerulus_mask != 0] = 0
    ifta_mask[artery_mask != 0] = 0
    ifta_mask[arteriole_mask != 0] = 0

    # Cortex/Medulla/Capsule post-processing
    # Cortex > Medulla > Capsule
    downsampled_glomerulus_mask = cv2.resize(glomerulus_mask, (cmc_X, cmc_Y), interpolation=cv2.INTER_NEAREST)
    cortex_mask[downsampled_glomerulus_mask != 0] = 255
    capsule_mask[cortex_mask != 0] = 0
    medulla_mask[cortex_mask != 0] = 0
    capsule_mask[medulla_mask != 0] = 0

    # Size filtering
    cmc_to_aag_ratio = (aag_X / cmc_X) * (aag_Y / cmc_Y)
    area_filter(glomerulus_mask, 500)
    area_filter(artery_mask, 500)
    area_filter(arteriole_mask, 500)
    area_filter(cortex_mask, 1000 / cmc_to_aag_ratio)
    area_filter(medulla_mask, 1000 / cmc_to_aag_ratio)
    area_filter(capsule_mask, 1000 / cmc_to_aag_ratio)

    # Touching Glomeruli check
    print("Solving Touching Glomeruli problem...")
    glomerulus_mask = detect_and_solve_touching_objects(glomerulus_mask)
    area_filter(glomerulus_mask, 500)

    pp_elapsed_time = time.time() - start_pp_time
    print("Post-processing elapsed time: ", pp_elapsed_time)

    # Exporting post-processed masks
    start_exp_time = time.time()
    cv2.imwrite(cortex_mask_output_path, cortex_mask)
    cv2.imwrite(medulla_mask_output_path, medulla_mask)
    cv2.imwrite(capsule_mask_output_path, capsule_mask)
    cv2.imwrite(glomerulus_mask_output_path, glomerulus_mask)
    cv2.imwrite(artery_mask_output_path, artery_mask)
    cv2.imwrite(arteriole_mask_output_path, arteriole_mask)
    cv2.imwrite(ifta_mask_output_path, ifta_mask)
    exp_elapsed_time = time.time() - start_exp_time
    print("Export elapsed time: ", exp_elapsed_time)

    # Exporting color mask
    color_mask = np.zeros((aag_Y, aag_X, 3), dtype=np.uint8)
    color_mask[upscaled_bioptic_tissue_mask > 0] = COLOR_BIOPSY_TISSUE_TUPLE
    color_mask[upscaled_bioptic_tissue_mask == 0] = COLOR_BACKGROUND_TUPLE

    upscaled_cortex_mask = cv2.resize(cortex_mask, (aag_X, aag_Y), interpolation=cv2.INTER_NEAREST_EXACT)
    upscaled_medulla_mask = cv2.resize(medulla_mask, (aag_X, aag_Y), interpolation=cv2.INTER_NEAREST_EXACT)
    upscaled_capsule_mask = cv2.resize(capsule_mask, (aag_X, aag_Y), interpolation=cv2.INTER_NEAREST_EXACT)
    color_mask[upscaled_capsule_mask > 0] = COLOR_CAPSULE_OTHER_TUPLE
    color_mask[upscaled_medulla_mask > 0] = COLOR_MEDULLA_TUPLE
    color_mask[upscaled_cortex_mask > 0] = COLOR_CORTEX_TUPLE

    color_mask[glomerulus_mask > 0] = COLOR_GLOMERULUS_TUPLE
    color_mask[artery_mask > 0] = COLOR_ARTERY_TUPLE
    color_mask[arteriole_mask > 0] = COLOR_ARTERIOLE_TUPLE

    color_mask[ifta_mask > 0] = COLOR_IFTA_TUPLE

    color_mask_path = os.path.join(color_dir, f"{wsi_name}.tiff")
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(color_mask_path, color_mask)
