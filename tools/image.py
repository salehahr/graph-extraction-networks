import cv2
import numpy as np

marker_size = 3
bgr_black = (0, 0, 0)
bgr_white = (255, 255, 255)
bgr_blue = (255, 0, 0)
bgr_green = (0, 255, 0)
bgr_red = (0, 0, 255)
bgr_lilac = (189, 130, 188)
bgr_yellow = (0, 255, 255)

degree_colours = [
    None,
    bgr_white,  # one neighbour
    bgr_green,  # helper
    bgr_red,  # three neighbours
    bgr_blue,
    bgr_lilac,
    bgr_yellow,
]

node_type_colours = [None, bgr_blue, bgr_red, bgr_black, bgr_yellow]

colour_codes = {
    "node_pos": [None, bgr_white],
    "degrees": degree_colours,
    "node_types": node_type_colours,
}


def normalize_mask(mask):
    """Mask Normalisation
    Function that returns normalised mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask


def generate_outputs(graph, dim: int) -> dict:
    """
    Generates output matrices of the graph's node attributes.
    """
    matr_node_pos = np.zeros((dim, dim, 1)).astype(np.uint8)
    matr_node_degrees = np.zeros((dim, dim, 1)).astype(np.uint8)
    matr_node_types = np.zeros((dim, dim, 1)).astype(np.uint8)

    for i, (col, row) in enumerate(graph.positions):
        matr_node_pos[row][col] = 1
        matr_node_degrees[row][col] = graph.num_node_neighbours[i]
        matr_node_types[row][col] = graph.node_types[i]

    return {
        "node_pos": matr_node_pos,
        "degrees": matr_node_degrees,
        "node_types": matr_node_types,
    }


def classifier_preview(output_matrices: dict, img_skel: np.ndarray) -> dict:
    """
    Serves as a visual test to check the correctness of the output matrices.
    Takes the classifier matrices as an input and generates for each matrix
    the corresponding visualisation.
    """
    base_img = cv2.cvtColor(img_skel, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    data_dict = {
        k: {"matrix": v.squeeze(), "colours": colour_codes[k]}
        for k, v in output_matrices.items()
    }

    debug_images = {}
    for k, v in data_dict.items():
        img = draw_circles(base_img, v["matrix"], v["colours"])
        debug_images[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return debug_images


def draw_circles(
    base_img: np.ndarray, classifier_matrix: np.ndarray, colour_list: list
) -> np.ndarray:
    """
    Draws circles on the image colour coded according to the unique values
    in the classifier matrix.
    Returns BGR image.
    """
    img = base_img.copy()
    unique_vals = np.unique(classifier_matrix)[1:]

    for val in unique_vals:
        positions = np.argwhere(classifier_matrix == val)
        for (y, x) in positions:
            cv2.circle(img, (x, y), marker_size, colour_list[val], -1)

    return img
