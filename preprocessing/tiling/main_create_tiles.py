import os
import glob
import click
from .slide_lib import segment_tiling


def tile_slide_images(source_dir: str, source_list: list, save_dir: str, patch_size: int=256, step_size: int=256, mag: int=20, index: int=0) -> None:
    """
    Tile whole slide images stored in the .svs/.ndpi/.scn format at the desired magnification.
    
    Parameters
    ----------
    source_dir : str
        Path to the source slide image (.svs) directory.
    source_list : str
        Path to the source slide image list (.txt) to be processed.
    save_dir : str
        Path to the save directory.
    patch_size : int
        Patch size.
    step_size : int
        Step size.
    mag : int
        Magnification for patch extraction.

    Returns
    -------
    None
    """
    
    if source_list:
        slide_list = [line.rstrip('\n') for line in open(source_list)]
        os.remove(source_list)
        
    tile_save_dir = os.path.join(save_dir, 'tiles')
    mask_save_dir = os.path.join(save_dir, 'masks')
    stitch_save_dir = os.path.join(save_dir, 'stitches')

    directories = {'source': slide_list if source_list else glob.glob(source_dir),
                   'save_dir': save_dir,
                   'tile_save_dir': tile_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}
    
    for key, val in directories.items():
        if key == 'source':
                continue
        os.makedirs(val, exist_ok=True)
    
    total_time = segment_tiling(**directories, patch_size=patch_size, mag_level=mag, step_size= step_size, index=index)
    print(f"The average processing time for each slide is {total_time:.2f} seconds.")

@click.command()
@click.option('--source_dir', type=str, help='path to the source slide image (.svs) directory')
@click.option('--source_list', type=str, help='path to the source slide image (.svs) list to be processed')
@click.option('--save_dir', type=str, help='path to the save directory')
@click.option('--patch_size', type=int, default=256, help='patch size')
@click.option('--step_size', type=int, default=256, help='step size')
@click.option('--mag', type=int, default=20, help='magnification for patch extraction')
@click.option('--index', type=int, default=0)
def generate_tiles(source_dir, source_list, save_dir, patch_size, step_size, mag, index):
    tile_slide_images(source_dir, source_list, save_dir, patch_size, step_size, mag, index)


if __name__ == '__main__':
    generate_tiles()
