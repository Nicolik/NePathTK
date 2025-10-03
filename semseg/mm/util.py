import os
import time
import datetime

from PIL import Image
from mmseg.apis import MMSegInferencer

from definitions import ROOT_DIR
from semseg.mm.config import MODELS_DICT_MMSEG_AAG, MODELS_DICT_MMSEG_CMC, MODELS_DICT_MMSEG_IFTA
from semseg.tma.download import download_mmseg_model


def get_cmc_mm_weights(arch, enc_name=None, microscopy_type="paraffin"):
    base_root = os.path.join(ROOT_DIR, 'semseg', 'mm')
    weight_file = f'{MODELS_DICT_MMSEG_CMC[arch]}.pth'
    ckpt_file = os.path.join(base_root, 'weights', microscopy_type, weight_file)
    return ckpt_file


def get_cmc_mm_configs(arch, enc_name=None, use_tta=True, microscopy_type="paraffin"):
    base_root = os.path.join(ROOT_DIR, 'semseg', 'mm')
    if use_tta:
        config_file = f'{MODELS_DICT_MMSEG_CMC[arch]}_tta.py'
    else:
        config_file = f'{MODELS_DICT_MMSEG_CMC[arch]}.py'
    cfg_file = os.path.join(base_root, 'configs', microscopy_type, config_file)
    return cfg_file


def get_aag_mm_weights(arch, enc_name=None, microscopy_type="paraffin"):
    base_root = os.path.join(ROOT_DIR, 'semseg', 'mm')
    weight_file = f'{MODELS_DICT_MMSEG_AAG[arch]}.pth'
    ckpt_file = os.path.join(base_root, 'weights', microscopy_type, weight_file)
    return ckpt_file


def get_ifta_mm_weights(arch, enc_name=None, microscopy_type="paraffin"):
    base_root = os.path.join(ROOT_DIR, 'semseg', 'mm')
    weight_file = f'{MODELS_DICT_MMSEG_IFTA[arch]}.pth'
    ckpt_file = os.path.join(base_root, 'weights', microscopy_type, weight_file)
    return ckpt_file


def get_aag_mm_configs(arch, enc_name=None, use_tta=True, microscopy_type="paraffin"):
    base_root = os.path.join(ROOT_DIR, 'semseg', 'mm')
    tta_text = "_tta" if use_tta else ""
    config_file = f'{MODELS_DICT_MMSEG_AAG[arch]}{tta_text}.py'
    cfg_file = os.path.join(base_root, 'configs', microscopy_type, config_file)
    return cfg_file


def get_ifta_mm_configs(arch, enc_name=None, use_tta=True, microscopy_type="paraffin"):
    base_root = os.path.join(ROOT_DIR, 'semseg', 'mm')
    tta_text = "_tta" if use_tta else ""
    config_file = f'{MODELS_DICT_MMSEG_IFTA[arch]}{tta_text}.py'
    cfg_file = os.path.join(base_root, 'configs', microscopy_type, config_file)
    return cfg_file


def download_cmc_semseg_model(arch, enc_name=None, microscopy_type="paraffin"):
    return download_mmseg_model("cmc", arch, backbone=enc_name, microscopy_type=microscopy_type)


def download_aag_semseg_model(arch, enc_name=None, microscopy_type="paraffin"):
    return download_mmseg_model("aag", arch, backbone=enc_name, microscopy_type=microscopy_type)


def download_ifta_semseg_model(arch, enc_name=None, microscopy_type="paraffin"):
    return download_mmseg_model("ifta", arch, backbone=enc_name, microscopy_type=microscopy_type)


def tensor_inference(inferencer):
    def compute(x, **kwargs):
        results = inferencer(x, **kwargs)
        preds = results['predictions']
        if isinstance(preds, list):
            return [p.astype('uint8') for p in preds]
        else:
            return preds.astype('uint8')
    return compute


def get_cortex_medulla_capsule_mmsegmenter(arch="mask2former", enc_name=None, device="cuda", use_tta=True, microscopy_type="paraffin"):
    cmc_mm_weights_path = get_cmc_mm_weights(arch, enc_name, microscopy_type=microscopy_type)
    if not os.path.exists(cmc_mm_weights_path):
        cmc_segmenter_path = download_cmc_semseg_model(arch, enc_name, microscopy_type=microscopy_type)
    cmc_mm_config_path = get_cmc_mm_configs(arch, enc_name, use_tta=use_tta, microscopy_type=microscopy_type)
    inferencer = MMSegInferencer(model=cmc_mm_config_path, weights=cmc_mm_weights_path, device=device)
    print(inferencer.pipeline)
    return tensor_inference(inferencer)


def get_glomerulus_artery_arteriole_mmsegmenter(arch="mask2former", enc_name=None, device="cuda", use_tta=True, microscopy_type="paraffin"):
    aag_mm_weights_path = get_aag_mm_weights(arch, enc_name, microscopy_type=microscopy_type)
    if not os.path.exists(aag_mm_weights_path):
        aag_segmenter_path = download_aag_semseg_model(arch, enc_name, microscopy_type=microscopy_type)
    aag_mm_config_path = get_aag_mm_configs(arch, enc_name, use_tta=use_tta, microscopy_type=microscopy_type)
    inferencer = MMSegInferencer(model=aag_mm_config_path, weights=aag_mm_weights_path, device=device)
    return tensor_inference(inferencer)


def get_ifta_cortex_mmsegmenter(arch="mask2former", enc_name=None, device="cuda", use_tta=True, microscopy_type="paraffin"):
    ifta_mm_weights_path = get_ifta_mm_weights(arch, enc_name, microscopy_type=microscopy_type)
    if not os.path.exists(ifta_mm_weights_path):
        ifta_segmenter_path = download_ifta_semseg_model(arch, enc_name, microscopy_type=microscopy_type)
    ifta_mm_config_path = get_ifta_mm_configs(arch, enc_name, use_tta=use_tta, microscopy_type=microscopy_type)
    inferencer = MMSegInferencer(model=ifta_mm_config_path, weights=ifta_mm_weights_path, device=device)
    return tensor_inference(inferencer)


def run_segmentation_mm(mm_model, tile_dir, binary_dir, multiclass_dir, class_names, device='cuda', batch_size=4, slice_size=16):
    tiles = os.listdir(tile_dir)
    valid_ext = '.jpeg'
    tiles = [t for t in tiles if t.endswith(valid_ext)]
    tiles_paths = [os.path.join(tile_dir, t) for t in tiles]

    slices_images = [tiles_paths[i:i + slice_size] for i in range(0, len(tiles_paths), slice_size)]
    silces_tiles = [tiles[i:i + slice_size] for i in range(0, len(tiles_paths), slice_size)]

    for j, (slice_image, slice_tile) in enumerate(zip(slices_images, silces_tiles)):
        print(f"processing Batch {j + 1}/{len(slices_images)} -- len slice: {len(slice_image)}")
        start_time_batch = time.time()
        pred_img_array_list = mm_model(slice_image, batch_size=batch_size)
        elapsed_time_batch = time.time() - start_time_batch
        eta = elapsed_time_batch * (len(slices_images) - j - 1)
        formatted_eta = str(datetime.timedelta(seconds=eta)).split('.')[0]
        print(f"Elapsed Time Batch: {elapsed_time_batch:.4f}. ETA: {formatted_eta}")


        for i, tile_name_ext in enumerate(slice_tile):
            pred_img_array = pred_img_array_list[i]
            tile_name = tile_name_ext.split(valid_ext)[0]
            output_filename_multiclass = tile_name + ".png"
            output_path_multiclass = os.path.join(multiclass_dir, output_filename_multiclass)
            print(f'Saving multiclass mask to: {output_path_multiclass}')
            pred_img = Image.fromarray(pred_img_array)
            pred_img.save(output_path_multiclass)

            # Create a subdir for current tile in 'Binary'
            tile_subdir = os.path.join(binary_dir, tile_name)
            if not os.path.exists(tile_subdir):
                os.makedirs(tile_subdir)

            # Save binary masks in subdir
            for i, class_name in enumerate(class_names):
                binary_mask = (pred_img_array == i).astype('uint8') * 255
                binary_img = Image.fromarray(binary_mask)
                output_filename_binary = f"{class_name}.png"
                output_path_binary = os.path.join(tile_subdir, output_filename_binary)
                print(f'Saving {class_name} mask to: {output_path_binary}')
                binary_img.save(output_path_binary)
