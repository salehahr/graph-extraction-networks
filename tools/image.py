def normalize_mask(mask):
    """ Mask Normalisation
    Function that returns normalised mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask
