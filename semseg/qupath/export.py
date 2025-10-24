import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
from definitions import OPENSLIDE
from semseg.qupath.cc_analysis import multimask_cc_analysis
from semseg.qupath.mask2json import polygons2json

os.environ['PATH'] = OPENSLIDE + ";" + os.environ['PATH']

from prepare.config import DOWNSAMPLE_FACTOR_TISSUE, DOWNSAMPLE_FACTOR_CMC, DOWNSAMPLE_FACTOR_GAA, MAGNIFICATION
from semseg.qupath.mask2poly import multimask2polygon
from semseg.qupath.utils2 import wsi2qpname, polygons2qu


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi', type=str, help='wsi name without extension')
    parser.add_argument('--ext', type=str, help='wsi extension')
    parser.add_argument('--wsi-dir', type=str, help='path/to/wsi/dir')
    parser.add_argument('--pp-dir', type=str, help="Directory from where to load post-processing results")
    parser.add_argument('--undersampling-tissue', type=int, help='Desired Undersampling', default=DOWNSAMPLE_FACTOR_TISSUE)
    parser.add_argument('--undersampling-cmc', type=int, help='Desired Undersampling', default=DOWNSAMPLE_FACTOR_CMC)
    parser.add_argument('--undersampling-aag', type=int, help='Desired Undersampling', default=DOWNSAMPLE_FACTOR_GAA)
    parser.add_argument('--magnification', type=int, help='Maximum Magnification Available', default=MAGNIFICATION)
    parser.add_argument('--export-qupath', help='whether to export qupath project', action="store_true")
    parser.add_argument('--qupath-export-dir', type=str, help='path/to/qupath', required=False)
    parser.add_argument('--export-json', help='whether to export json annotations', action="store_true")
    parser.add_argument('--base-export-dir', type=str, help='path/to/export', required=False)
    args = parser.parse_args()

    wsi_name = args.wsi
    ext = args.ext
    post_processed_mask_dir = args.pp_dir
    wsi_dir = args.wsi_dir
    undersampling_bt = args.undersampling_tissue
    undersampling_cmc = args.undersampling_cmc
    undersampling_aag = args.undersampling_aag
    undersampling_ifta = undersampling_aag
    magnification = args.magnification
    qupath_export_dir = args.qupath_export_dir
    export_dir = args.base_export_dir
    export_json = args.export_json
    export_qupath = args.export_qupath

    poly2json_export_dir = os.path.join(export_dir, 'poly2json-output')
    os.makedirs(poly2json_export_dir, exist_ok=True)

    # Input masks paths
    cortex_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_Cortex_mask.png")
    medulla_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_Medulla_mask.png")
    capsule_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_CapsuleOther_mask.png")
    glomerulus_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_Glomerulus_mask.png")
    arteriole_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_Arteriole_mask.png")
    artery_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_Artery_mask.png")
    ifta_mask_path = os.path.join(post_processed_mask_dir, f"{wsi_name}_IFTACortex_mask.png")

    # Cortex Medulla Capsule
    cortex_mask = cv2.imread(cortex_mask_path, cv2.IMREAD_GRAYSCALE)
    medulla_mask = cv2.imread(medulla_mask_path, cv2.IMREAD_GRAYSCALE)
    capsule_mask = cv2.imread(capsule_mask_path, cv2.IMREAD_GRAYSCALE)
    cmc_Y, cmc_X = cortex_mask.shape

    # Glomerulus Artery Arteriole
    glomerulus_mask = cv2.imread(glomerulus_mask_path, cv2.IMREAD_GRAYSCALE)
    arteriole_mask = cv2.imread(arteriole_mask_path, cv2.IMREAD_GRAYSCALE)
    artery_mask = cv2.imread(artery_mask_path, cv2.IMREAD_GRAYSCALE)
    aag_Y, aag_X = glomerulus_mask.shape

    # IFTACortex
    ifta_mask = cv2.imread(ifta_mask_path, cv2.IMREAD_GRAYSCALE)
    ifta_Y, ifta_X = ifta_mask.shape

    class_masks_cmc = {
        'Cortex': cortex_mask,
        'Medulla': medulla_mask,
        'CapsuleOther': capsule_mask,
    }
    class_masks_aag = {
        'Glomerulus': glomerulus_mask,
        'Artery': artery_mask,
        'Arteriole': arteriole_mask,
    }
    class_masks_ifta = {
        'IFTACortex': ifta_mask,
    }

    # Convert masks to polygons
    polygons_dict_all = {}
    for class_masks, undersampling in zip([class_masks_cmc, class_masks_aag, class_masks_ifta],
                                          [undersampling_cmc, undersampling_aag, undersampling_ifta]):
        class_masks = multimask_cc_analysis(class_masks, magnification, undersampling)
        polygons_dict = multimask2polygon(class_masks, undersampling)
        polygons_dict_all.update(polygons_dict)

    # Export polygons in QuPath
    wsi_qpname, wsi_name_ext, wsi_name_idx, wsi_name_without_idx = wsi2qpname(wsi_name, ext)
    if export_qupath:
        print(f"Exporting qupath to {qupath_export_dir}...")
        polygons2qu(wsi_qpname, polygons_dict_all, qupath_export_dir)
    if export_json:
        print(f"Exporting JSON to {poly2json_export_dir}")
        polygons2json(polygons_dict_all, wsi_qpname, wsi_dir, poly2json_export_dir)
