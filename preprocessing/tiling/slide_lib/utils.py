
import os
import cv2
import openslide
import numpy as np
import pandas as pd
import typing as tp
from typing import Union
import skimage.filters as sk_filters
from PIL import Image
from .constants_color import PENS_RGB
import multiprocessing as mp
import concurrent.futures


def filter_contours(contours: list, hierarchy: np.array, filter_params: dict)->tuple[list, list]:
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
    all_holes = []
    
    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: 'continue'
        if tuple((filter_params['a_t'],)) < tuple((a,)): 
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]
    
    hole_contours = []
    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids ]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []
        
        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)
    return foreground_contours, hole_contours


def move_small(mask: np.array)->np.array:
    kernel = np.ones((10,10),np.uint8)
    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return result


def hysteresis_threshold(gray: np.array, low: int | float=20, high: int | float=50)->np.array:
    result = sk_filters.apply_hysteresis_threshold(gray, low, high)
    return result


def gray_filter(rgb: np.array, tolerance: int | float=23)->np.array:
    rgb = rgb.astype(np.int32)
    rg_diff = (abs(rgb[:, :, 0].mean() - rgb[:, :, 1].mean()) <= tolerance)
    rb_diff = (abs(rgb[:, :, 0].mean() - rgb[:, :, 2].mean()) <= tolerance)
    gb_diff = (abs(rgb[:, :, 1].mean() - rgb[:, :, 2].mean()) <= tolerance)
    result = ~(rg_diff & rb_diff & gb_diff)
    return result


def get_gradient_magnitude(im: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient magnitude of a given image.
    
    Parameters
    ----------
    im : numpy array
        The image for which to calculate the gradient magnitude.

    Returns
    -------
    numpy array
        The gradient magnitude of the image.
    """
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dx_abs = cv2.convertScaleAbs(dx)
    dy_abs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dx_abs, 0.5, dy_abs, 0.5, 0)

    return mag


def get_binary_closing(im: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """
    Perform binary closing operation on a given image.
    
    Parameters
    ----------
    im : numpy array
        The image to perform the operation on.
    kernel_size : int, optional
        The size of the kernel used for the morphological operation.

    Returns
    -------
    numpy array
        The image after binary closing operation.
    """
    mag_binary = (im > 20).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    closing = cv2.morphologyEx(mag_binary, cv2.MORPH_CLOSE, kernel)
    return closing


def get_magnitude_closing(im: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """
    Perform binary closing operation on the gradient magnitude of a given image.
    
    Parameters
    ----------
    im : numpy array
        The image to perform the operation on.
    kernel_size : int, optional
        The size of the kernel used for the morphological operation.

    Returns
    -------
    numpy array
        The image after binary closing operation.
    """
    mag = get_gradient_magnitude(im)
    closing = get_binary_closing(mag, kernel_size)
    return closing


def scaleContourDim(contours: list, scale: int | float)->list:
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def scaleHolesDim(contours: list, scale: int | float)->list:
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]


class Contour_Checking_fn(object):
	# Defining __call__ method 
	def __call__(self, pt): 
		raise NotImplementedError


class isInContour(Contour_Checking_fn):
	def __init__(self, contour: list, patch_size: int | float, center_shift: int | float=0.5)->None:
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt: np.array)->int: 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0


def isInHoles(holes: list, pt: np.array, patch_size: int|float)->int:
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
            return 1
    
    return 0


def isInContours(cont_check_fn: tp.Callable[[np.array], int], pt: np.array, holes: list=None, patch_size: int|float=256)->int:
    if cont_check_fn(pt):
        if holes:
            return not isInHoles(holes, pt, patch_size)
        else:
            return 1
    return 0


def process_coord_candidate(coord: np.array, contour_holes: list, ref_patch_size: int|float, cont_check_fn: tp.Callable[[np.array], int])->np.array:
    if isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
        return coord
    else:
        return None

def isWhitePatch(patch: np.array, thresh: int|float=230)->bool:
    return True if patch.mean() > thresh else False


def isBlackPatch(patch: np.array, thresh: int|float=50)->bool:
    return True if patch.mean() < thresh else False


def tileWriter(wsi: openslide.OpenSlide, coord: Union[np.array, tuple], attr_dict: dict)->bool:
    patch = wsi.read_region(coord, attr_dict['patch_level'], 
                            (int(attr_dict['patch_size']*attr_dict['downsample']), int(attr_dict['patch_size']*attr_dict['downsample']))).convert('RGB')
    patch = np.array(patch.resize((attr_dict['patch_size'], attr_dict['patch_size'])))
    # if not isWhitePatch(patch) and  and get_magnitude_closing(patch).mean() > 0.35 and gray_filter(patch):
    if not isBlackPatch(patch) and get_magnitude_closing(patch).mean() > 0.35:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(attr_dict['save_path'], attr_dict['name'], f"{attr_dict['name']}_x_y_{coord[0]}_{coord[1]}.jpg"), patch)
        return coord
    else:
        return None


def save_tiles(slide_path: str, asset_dict: dict, attr_dict: dict)->np.array:
    wsi = openslide.open_slide(slide_path)
    coords = asset_dict['coords']
    coords = coords[1:] if np.array_equal(coords[0], np.array([0, 0])) else coords
    num_workers = mp.cpu_count()
    num_workers = (num_workers / 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(tileWriter, [wsi]*len(coords), coords, [attr_dict]*len(coords))
    filtered_coords = []
    for item in results:
        if item is not None:
            filtered_coords.append(item)
    return np.array(filtered_coords)


def DrawGrid(img: np.array, coord: np.array, shape: int|float, thickness: int|float=2, color:tuple=(0,0,0,255))->np.array:
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), color, thickness=thickness)
    return img


def DrawMapFromCoords(canvas: np.array, wsi: openslide.OpenSlide, coords: np.array, patch_size: int|float, vis_level: int|float)->Image.Image:
    downsamples = wsi.level_downsamples[vis_level]
    indices = np.arange(len(coords))
    total = len(indices)
    patch_size = int(np.ceil(patch_size/downsamples))
    print(f'downscaled patch size: {patch_size}x{patch_size}')

    for idx in range(total):
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi.read_region(coord, vis_level, (patch_size, patch_size)).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size, coord[0]:coord[0]+patch_size, :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size, coord[0]:coord[0]+patch_size, :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


def write_tile_info(tile_df: pd.DataFrame, results: np.array, slide_id: str, tile_save_dir: str, slide_annot: pd.DataFrame) -> pd.DataFrame:
    tile_annot = [slide_annot[slide_annot['slide'] == slide_id]['family'].values[0]] * len(results)
    file_path = [f"{slide_id}_x_y_{coord[0]}_{coord[1]}.jpg" for coord in results]
    slide_info = [slide_id] * len(results)
    tile_df = pd.concat([tile_df, pd.DataFrame({'file_path': file_path, 'slide': slide_info, 'family': tile_annot})], ignore_index=True)
    return tile_df


def pen_filter(bands: np.array, pen_color: str)->np.array:
    r, g, b = bands[:, :, 0], bands[:, :, 1], bands[:, :, 2]
    thresholds = PENS_RGB[pen_color]

    if pen_color == "red":
        t = thresholds[0]
        mask = (r > t[0]) & (g < t[1]) & (b < t[2])

        for t in thresholds[1:]:
            mask = mask | ((r > t[0]) & (g < t[1]) & (b < t[2]))

    elif pen_color == "green":
        t = thresholds[0]
        mask = (r < t[0]) & (g > t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g > t[1]) & (b > t[2])

    elif pen_color == "blue":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g < t[1]) & (b > t[2])

    else:
        raise Exception(f"Error: pen_color='{pen_color}' not supported")

    return mask


def find_best_seg_level(wsi, width: int)->int:
    level_dims = wsi.level_dimensions
    abs_ls = []
    for idx, dim in enumerate(level_dims):
        abs_ls.append(abs(dim[0] - width))
    best_level = np.argmin(abs_ls)
    
    return best_level

