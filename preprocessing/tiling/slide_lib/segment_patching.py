import os
import cv2
import time
import openslide
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing as mp
from .utils import *


def segment(wsi: openslide.OpenSlide)->tuple[list, list, Image.Image, float]:
    start_time = time.time()

    try:
        seg_level = find_best_seg_level(wsi, 1024) # 1024 is the reference size for segmentation
    except:
        seg_level = -1

    img = np.array(wsi.read_region((0, 0), seg_level, wsi.level_dimensions[seg_level]).convert('RGB'))  # doing segmentation at the mag level closest to 1024x1024 
    img_gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale and invert
    # r_pen = pen_filter(img, 'red')
    g_pen = pen_filter(img, 'green')
    b_pen = pen_filter(img, 'blue')
    pen_mask = ~(cv2.dilate(get_binary_closing((g_pen | b_pen).astype(int)*255, 13)*255, np.ones((7, 7), np.uint8), iterations=1).astype(bool))
    bw_1 = get_binary_closing(hysteresis_threshold(img_gray).astype(np.uint8)*255, 13)
    bw = move_small((bw_1 & pen_mask)).astype(np.uint8)
    scale = wsi.level_downsamples[seg_level]
    scaled_ref_patch_area = int(256 ** 2 / scale ** 2)

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    foreground_contours, hole_contours = filter_contours(contours, hierarchy, {'a_t':0.2 * scaled_ref_patch_area, 'a_h':  scaled_ref_patch_area, 'max_n_holes':10})
    
    # save mask file
    line_thickness = int(100 / scale)
    cv2.drawContours(img, foreground_contours, -1, (0, 255, 0), line_thickness, lineType=cv2.LINE_8)
    img = Image.fromarray(img)

    contours_tissue = scaleContourDim(foreground_contours, scale)
    holes_tissue = scaleHolesDim(hole_contours, scale)
    contours_tissue = [contours_tissue[i] for i in range(len(contours_tissue))]  # coordinates at the highest resolution
    holes_tissue = [holes_tissue[i] for i in range(len(contours_tissue))]  # coordinates at the highest resolution

    return contours_tissue, holes_tissue, img, time.time() - start_time


def patching(wsi: openslide.OpenSlide, contours: list, holes: list, tile_save_dir: str, 
             patch_size: float | int, mag_level: float | int, step_size: float | int, slide_path: str)->tuple[list, float, pd.DataFrame]:
    start_time = time.time()
    
    highest_mag = int(wsi.properties['openslide.objective-power'])
    highest_downsample = int(highest_mag/mag_level)  # downsample factor for the highest magnification
    patch_level = wsi.get_best_level_for_downsample(highest_downsample)  # find the best level for tiling
    best_downsample = highest_downsample / (wsi.level_downsamples[patch_level]/wsi.level_downsamples[0]) # downsample factor for the best level
    ref_patch_size = int(patch_size * highest_downsample)  # patch size at the highest magnification
    ref_step_size = int(step_size * highest_downsample)  # step size at the highest magnification

    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    coord_record = []

    for cont_idx, cont in enumerate(contours):
        start_x, start_y, w, h = cv2.boundingRect(cont)
        stop_x = start_x + w
        stop_y = start_y + h
        # print("Bounding Box:", start_x, start_y, w, h)
        # print("Contour Area:", cv2.contourArea(cont))

        cont_check_fn = isInContour(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
        x_range = np.arange(start_x, stop_x, step=ref_step_size)
        y_range = np.arange(start_y, stop_y, step=ref_step_size)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        # multiprocessing
        num_workers = mp.cpu_count()
        num_workers = int(num_workers / 2)  # use half of the available cores (could be adjusted based on the memory usage)
        pool = mp.Pool(processes=num_workers)
        iterable = [(coord, holes[cont_idx], ref_patch_size, cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(process_coord_candidate, iterable)
        pool.close()

        results = np.array([result for result in results if result is not None])
        print('Extracted {} points within the contour'.format(len(results)))

        if len(results)>1:
            asset_dict = {'coords': results}
            attr = {'patch_size' :            patch_size, # patch size at the tiling level
                    'mag_level' :             mag_level, # magnification level for tiling
                    'patch_level':            patch_level, # the best level for tiling (1/4/...)
                    'downsample':             best_downsample, # downsample factor for the best level
                    'level_dim':              (wsi.level_dimensions[0][0]/highest_downsample, wsi.level_dimensions[0][1]/highest_downsample), # dimensions at the tiling level
                    'name':                   slide_id,
                    'save_path':              tile_save_dir}

            attr_dict = {'coords': attr}
            os.makedirs(os.path.join(tile_save_dir, slide_id), exist_ok=True)
            final_coords = save_tiles(slide_path, asset_dict, attr_dict['coords'])
            coord_record.extend(final_coords)

    return coord_record, time.time() - start_time


def stitching(wsi: openslide.OpenSlide, coords: list, patch_size: float | int, mag_level: float | int, step_size: float | int, 
              downscale: float | int = 64)->tuple[Image.Image, float]:
    start_time = time.time()

    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    ori_w, ori_h = wsi.level_dimensions[0]
    highest_patch_size = patch_size*int(wsi.properties["openslide.objective-power"])/mag_level

    print('Start stitching...')
    print(f'Original size: w: {ori_w} x h: {ori_h}')
    print(f'Downscaled size for stiching: w: {w} x h: {h}')
    print(f'Number of patches: {len(coords)}')
    print(f'Patch size: {patch_size}x{patch_size} at magnification: {mag_level}')
    print(f'Ref patch size: {highest_patch_size}')

    heatmap = Image.new(size=(w, h), mode="RGB", color=(0, 0, 0))
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi, coords, highest_patch_size, vis_level)

    return heatmap, time.time() - start_time


def segment_tiling(source: str, save_dir: str, tile_save_dir: str, mask_save_dir: str, stitch_save_dir: str,
                   patch_size: int | float =256, mag_level: int | float =20, step_size: int | float =256, index: int = 0)->None:
    slides = source
    df = pd.DataFrame({"slide_path": slides, "slide_mpp": np.nan, "slide_mag": np.nan})

    seg_time = 0.
    tile_time = 0.
    stitch_time = 0.
    for i in range(len(df)):
        slide_path = df['slide_path'][i]
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        print(f"\nprogress: {i+1}/{len(df)}")
        print(f"processing {slide_name}")
        
        try:
            wsi = openslide.open_slide(slide_path)

            try:
                df.loc[i, 'slide_mpp'] = float(wsi.properties['aperio.MPP'])
            except (KeyError, ValueError):
                try:
                    df.loc[i, 'slide_mpp'] = float(wsi.properties['openslide.mpp-y'])
                except (KeyError, ValueError):
                    df.loc[i, 'slide_mpp'] = 'NA'

            try:
                if 'openslide.objective-power' in wsi.properties:
                    df.loc[i, 'slide_mag'] = float(wsi.properties['openslide.objective-power'])
                else:
                    df.loc[i, 'slide_mag'] = float(wsi.properties['aperio.AppMag'])
            except (KeyError, ValueError):
                df.loc[i, 'slide_mag'] = 'NA'
                print(f"slide {slide_name} has no magnification information")
                continue

            contour_coord, hole_coord, mask, seg_time_elapsed = segment(wsi)
            mask.save(os.path.join(mask_save_dir, f'{slide_name}.jpg'))

            coords, tile_time_elapsed = patching(wsi, contour_coord, hole_coord, tile_save_dir, patch_size, mag_level, step_size, slide_path)

            heatmap, stitch_time_elapsed = stitching(wsi, coords, patch_size, mag_level, step_size, downscale=64)
            heatmap.save(os.path.join(stitch_save_dir, slide_name+'.jpg'))

            print(f"segmentation took {seg_time_elapsed:.2f} seconds")
            print(f"patching took {tile_time_elapsed:.2f} seconds")
            print(f"stitching took {stitch_time_elapsed:.2f} seconds")
            seg_time += seg_time_elapsed
            tile_time += tile_time_elapsed
            stitch_time += stitch_time_elapsed
        
        except openslide.OpenSlideError:
            with open(os.path.join(save_dir, 'error_slides.txt'), 'a') as f:
                f.write(df['slide_path'][i]+'\n')
    
    os.makedirs(os.path.join(save_dir, 'slide_info'), exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'slide_info', f'slide_info_{index}.csv'), index=False)
    print(f"\nslide info (mpp, magnification) saved to {os.path.join(save_dir, f'slide_info_{index}.csv')}")
    seg_time /= len(df)
    tile_time /= len(df)
    stitch_time /= len(df)
    print(f"average segmentation time in s per slide: {seg_time:.2f}")
    print(f"average patching time in s per slide: {tile_time:.2f}")
    print(f"average stiching time in s per slide: {stitch_time:.2f}")

    return seg_time+tile_time+stitch_time
        


