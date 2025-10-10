OP_EPS = 0.10
OPENSLIDE_EXTENSIONS = ('.ndpi', '.svs', '.mrxs', '.tif', '.tiff')
BIOFORMATS_EXTENSIONS = ('.scn', '.czi', '.ome.tif', '.ome.tiff')
MULTI_IMAGE_EXTENSIONS = ('.scn', '.czi')


def check_desired_op(closest_op, desired_op):
    return abs((closest_op - desired_op) / desired_op) < OP_EPS
