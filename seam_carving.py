import numpy as np

def seam_carve(image: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """
    Removes a seam from the image depending on the seam mask. Returns an image that has one column less than image.
    """
    shrunken: np.ndarray = image[seam_mask].reshape((image.shape[0], -1, image.shape[-1]))

    return shrunken.squeeze()


def update_global_mask(global_mask: np.ndarray, new_mask: np.ndarray) -> np.ndarray:
    """
    Updates the global_mask (bool) that contains all previous seams by adding the new path contained in new_mask.
    """
    reduced_idc: np.ndarray = np.indices(global_mask.shape)[:, ~global_mask][:, new_mask.flat]
    seam_mask: np.ndarray = np.ones_like(global_mask, dtype=bool)
    seam_mask[reduced_idc[0], reduced_idc[1]] = False

    return seam_mask


def calculate_accum_energy(energy: np.ndarray) -> np.ndarray:
    """
    Function computes the accumulated energies.
    """
    energy_accumulated: np.ndarray = np.zeros_like(energy)

    for x in range(energy.shape[0]):
        for y in range(energy.shape[1]):
            if x == 0:
                energy_accumulated[0, y] = energy[0, y]
            else:
                energy_accumulated[x, y] = energy[x, y]
                if y == 0:  # left edge
                    energy_accumulated[x, y] += min(energy_accumulated[x - 1, y], energy_accumulated[x - 1, y + 1])
                elif y == energy.shape[1] - 1:  # right edge
                    energy_accumulated[x, y] += min(energy_accumulated[x - 1, y - 1], energy_accumulated[x - 1, y])
                else:
                    energy_accumulated[x, y] += min(energy_accumulated[x - 1, y - 1], energy_accumulated[x - 1, y], energy_accumulated[x - 1, y + 1])

    return energy_accumulated


def create_seam_mask(energy: np.ndarray) -> np.ndarray:
    """
    Creates and returns a boolean matrix containing zeros (False) where to remove the seam.
    """
    mask_bool: np.ndarray = np.ones_like(energy, dtype=bool)

    start_index: int = np.argmin(energy[-1])

    mask_bool[energy.shape[0] - 1, start_index] = False

    index: int = start_index
    for row in reversed(range(energy.shape[0] - 1)):
        if index == 0:  # left edge
            children = np.array([0, energy[row, index], energy[row, index + 1]])
        elif index == energy.shape[1] - 1:  # right edge
            children = np.array([energy[row, index - 1], energy[row, index]])
        else:
            children = np.array([energy[row, index - 1], energy[row, index], energy[row, index + 1]])
        mask_bool[row, index] = False
        index += np.argmin(children) - 1

    return mask_bool