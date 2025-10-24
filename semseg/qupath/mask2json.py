import logging
import os
import json
import numpy as np
import pandas as pd
from detection.qupath.defs import CATEGORY_ID, MIN_AREA_BBOX_JSON
from detection.qupath.local.config_mescnn import MIN_AREA_BBOX_GLOMERULUS
from detection.qupath.utils import init_data_dict
from detection.qupath.paths import path_image_data
from semseg.qupath.mask2poly import multimask2polygon


def mask2json(class_masks, undersampling, id_name, wsi_dir, export_dir=None):
    polygons_dict = multimask2polygon(class_masks, undersampling)
    return polygons2json(polygons_dict, id_name, wsi_dir, export_dir=export_dir)


def polygons2json(polygons_dict, id_name, wsi_dir, export_dir=None):
    images_list_dict = []
    dict_rois = init_data_dict()

    if export_dir:
        json_dataset = os.path.join(export_dir, f'{id_name} dataset_detectron2.json')
        csv_rois = os.path.join(export_dir, f'{id_name} rois.csv')

        if os.path.exists(json_dataset) and os.path.exists(csv_rois):
            with open(json_dataset, 'r') as fp:
                images_list_dict = json.load(fp)
            df_rois = pd.read_csv(csv_rois)
            dict_rois = {str(key): list(df_rois[key]) for key in df_rois}

    s, ext, id_name, path_to_image = path_image_data(id_name, wsi_dir, add_dir_mrxs=True)

    for shape_object_type in ['Glomerulus', 'Artery', 'Arteriole']:
        tile_offset_x = 100
        tile_offset_y = 100
        for shape_object in polygons_dict[shape_object_type]:
            xg, yg = shape_object.exterior.coords.xy
            xg, yg = np.array(xg, dtype=np.int32), np.array(yg, dtype=np.int32)
            if xg.size > 0 and yg.size > 0:
                xt, yt = xg.min() - tile_offset_x, yg.min() - tile_offset_y
                wt, ht = xg.max() - xg.min() + 2 * tile_offset_x, yg.max() - yg.min() + 2 * tile_offset_y
                tile_id = f"{shape_object_type} {id_name} [{xt}, {yt}, {wt}, {ht}]"
                tile_filename = f"{tile_id}.jpeg"
                print(f"Tile: [{xt}, {yt}, {wt}, {ht}]")

                xg, yg = xg - xt, yg - yt

                bbox_g = [int(xg.min()), int(yg.min()), int(xg.max()), int(yg.max())]
                polygon_shape_object = []
                for xgi, ygi in zip(xg, yg):
                    polygon_shape_object.extend([int(xgi), int(ygi)])

                area_bbox = (yg.max() - yg.min()) * (xg.max() - xg.min())
                if area_bbox > MIN_AREA_BBOX_JSON[shape_object_type]:
                    dict_rois['image-id'].append(id_name)
                    dict_rois['filename'].append(tile_filename)
                    dict_rois['path-to-wsi'].append(path_to_image)
                    dict_rois['ext'].append(ext)
                    dict_rois['s'].append(s)
                    dict_rois['x'].append(xt)
                    dict_rois['y'].append(yt)
                    dict_rois['w'].append(wt)
                    dict_rois['h'].append(ht)

                    annotations_tile_shape_object = {
                        'bbox': bbox_g,
                        'bbox_mode': 'BoxModeXYXY_ABS',
                        'category_id': int(CATEGORY_ID[shape_object_type]),
                        'segmentation': [polygon_shape_object]
                    }

                    image_dict = {
                        'file_name': tile_filename,
                        'height': int(ht),
                        'width': int(wt),
                        'image_id': tile_id,
                        'annotations': [annotations_tile_shape_object]
                    }
                    images_list_dict.append(image_dict)
                else:
                    logging.warning(f"image_id: {tile_id}, found area ({area_bbox})"
                                    f" lesser than {MIN_AREA_BBOX_GLOMERULUS}!")

    if export_dir:
        with open(json_dataset, 'w') as fp:
            json.dump(images_list_dict, fp)

        df_rois = pd.DataFrame(data=dict_rois)
        df_rois.to_csv(csv_rois, index=False)

    return images_list_dict
