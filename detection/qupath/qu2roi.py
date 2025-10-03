import json
import os.path
import numpy as np
import shapely
from paquo.projects import QuPathProject

from semseg.qupath.utils import get_wsi_name

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert QuPathProject Annotations to ROI in JSON format')
    parser.add_argument('-e', '--tissue-tiler-dir', type=str, help='path/to/export', required=True)
    parser.add_argument('-w', '--wsi-dir', type=str, help='path/to/wsi/dir', required=True)
    parser.add_argument('-q', '--qupath', type=str, help='path/to/qupath', required=True)

    args = parser.parse_args()
    path_to_wsi = args.wsi_dir
    tissue_tiler_dir = args.tissue_tiler_dir
    path_to_qupath = args.qupath

    wsi_list = []
    shapes_list = []

    with QuPathProject(path_to_qupath, mode='r') as qp:
        num_images = len(qp.images)
        print(f"opened project '{qp.name}' with {num_images} images")
        for i, image in enumerate(qp.images):
            image_name = image.image_name
            wsi_list.append(image_name)
            bbox_dict_list = []
            shapes = None

            print(f"Processing image {image_name}, trying to load annotations...")
            if image.hierarchy and image.hierarchy.annotations:
                annotations = image.hierarchy.annotations
                shapes = {
                    'AutoSegment': [],
                }

                print(f"Image {image_name} has {len(annotations)} annotations.")
                for a, annotation in enumerate(annotations):

                    # annotations are paquo.pathobjects.QuPathPathAnnotationObject instances
                    # their ROIs are accessible as shapely geometries via the .roi property
                    name = annotation.path_class.name if annotation.path_class else "none"
                    if name in shapes:
                        print(f"> [I = {(i + 1):3d}/{len(qp.images):3d}] "
                              f"[A = {(a + 1):3d}/{len(annotations):3d}] class: {name}")
                        if type(annotation.roi) == shapely.geometry.polygon.Polygon:
                            shapes[name].append(annotation.roi)

                shapes_list.append(shapes)

                for shape_object in shapes['AutoSegment']:
                    xg, yg = shape_object.exterior.coords.xy
                    xg, yg = np.array(xg, dtype=np.int32), np.array(yg, dtype=np.int32)
                    xmin, ymin = xg.min(), yg.min()
                    xmax, ymax = xg.max(), yg.max()
                    xt, yt = xmin, ymin
                    wt, ht = xmax - xmin, ymax - ymin

                    print(f"AutoSegment ROI: ({xt}, {yt}, {wt}, {ht})")
                    bbox_dict = {"top_left": [int(xmin), int(ymin)], "bottom_right": [int(xmax), int(ymax)]}
                    bbox_dict_list.append(bbox_dict)

                print(f"Image {image_name} has {len(bbox_dict_list)} ROIs")
                json_name = get_wsi_name(image_name) + '.json'
                json_file = os.path.join(tissue_tiler_dir, json_name)

                if os.path.exists(json_file):
                    with open(json_file, 'r') as fp:
                        dict_metadata = json.load(fp)
                    dict_metadata["bounding_boxes"] = bbox_dict_list
                    with open(json_file, 'w') as fp:
                        json.dump(dict_metadata, fp)
                        print('[qu2roi] bbox saved')
