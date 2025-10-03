import os
from xml.dom import minidom
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from shapely import Polygon
from detection.qupath.pkl2qu import get_or_create_entry
from semseg.qupath.utils import (category_colors, COLOR_BIOPSY_TISSUE,
                                 COLOR_CORTEX, COLOR_MEDULLA, COLOR_CAPSULE_OTHER,
                                 COLOR_GLOMERULUS, COLOR_ARTERY, COLOR_ARTERIOLE,
                                 COLOR_IFTA,
                                 point_from_pixels_to_physical,)


def wsi2qpname(wsi_name, ext):
    if ext == ".scn":
        wsi_name_without_idx = "_".join(wsi_name.split("_")[:-1])
        wsi_name_idx = int(wsi_name.split('_')[-1])
        if wsi_name_idx >= 1:
            wsi_qpname = f"{wsi_name_without_idx}{ext} - Series {wsi_name_idx}"
        else:
            wsi_qpname = f"{wsi_name_without_idx}{ext} - macro"
        wsi_name_ext = f"{wsi_name_without_idx}{ext}"
    elif ext == '.czi':
        wsi_name_without_idx = "_".join(wsi_name.split("_")[:-1])
        wsi_name_idx = int(wsi_name.split('_')[-1]) + 1
        wsi_qpname = f"{wsi_name_without_idx}{ext} - Scene #{wsi_name_idx}"
        wsi_name_ext = f"{wsi_name_without_idx}{ext}"
    else:
        wsi_name_without_idx = wsi_name
        wsi_qpname = f"{wsi_name}{ext}"
        wsi_name_ext = f"{wsi_name}{ext}"
        wsi_name_idx = ""
    return wsi_qpname, wsi_name_ext, wsi_name_idx, wsi_name_without_idx


def polygons2qu(path_to_wsi, polygons_dict, qupath_project_dir, only_existing_entries=False):
    from paquo.projects import QuPathProject
    from paquo.classes import QuPathPathClass

    print(f"[polygons2qu] ---- Start")

    with QuPathProject(qupath_project_dir, mode='a') as qp:
        print(f"[polygons2qu] Created Project {qp.name}!")

        new_classes = {
            "BiopsyTissue": QuPathPathClass(name="BiopsyTissue", color=COLOR_BIOPSY_TISSUE),
            "Cortex": QuPathPathClass(name="Cortex", color=COLOR_CORTEX),
            "Medulla": QuPathPathClass(name="Medulla", color=COLOR_MEDULLA),
            "CapsuleOther": QuPathPathClass(name="CapsuleOther", color=COLOR_CAPSULE_OTHER),
            "Glomerulus": QuPathPathClass(name="Glomerulus", color=COLOR_GLOMERULUS),
            "Artery": QuPathPathClass(name="Artery", color=COLOR_ARTERY),
            "Arteriole": QuPathPathClass(name="Arteriole", color=COLOR_ARTERIOLE),
            "IFTACortex": QuPathPathClass(name="IFTACortex", color=COLOR_IFTA)
        }

        # Adding new classes to QuPath Project
        qp.path_classes = [new_classes[key] for key in new_classes]

        print(f"[polygons2qu] path_to_wsi: {path_to_wsi}")
        entry = get_or_create_entry(qp, path_to_wsi, only_existing_entries=only_existing_entries)
        print(f"[polygons2qu] type(entry): {type(entry)}")

        # Adding the annotations
        for class_name, polygons in polygons_dict.items():
            for i, poly_coord in enumerate(polygons):
                polygon = Polygon(poly_coord)
                print(f"[polygons2qu] Adding annotation: {new_classes[class_name]}")
                entry.hierarchy.add_annotation(roi=polygon, path_class=new_classes[class_name])
        print(f"[polygons2qu] done. Please look at {qp.name} in QuPath.")

    print(f"[polygons2qu] ---- End")
