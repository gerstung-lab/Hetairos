import os
import glob
import click
from hetairos.preprocessing.tiling.slide_lib import segment_tiling


SUPPORTED_SLIDE_EXTENSIONS = ('.svs', '.ndpi', '.scn')


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

    if bool(source_dir) == bool(source_list):
        raise ValueError("Provide exactly one of source_dir or source_list.")

    if source_list:
        slide_list = [line.rstrip('\n') for line in open(source_list) if line.strip()]
        # Bug fix E: removed os.remove(source_list) — the caller owns the file
    else:
        slide_list = []
        for ext in SUPPORTED_SLIDE_EXTENSIONS:
            slide_list.extend(glob.glob(os.path.join(source_dir, f'*{ext}')))
        slide_list = sorted(slide_list)

    tile_save_dir = os.path.join(save_dir, 'tiles')
    mask_save_dir = os.path.join(save_dir, 'masks')
    stitch_save_dir = os.path.join(save_dir, 'stitches')

    directories = {'source': slide_list,
                   'save_dir': save_dir,
                   'tile_save_dir': tile_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        if key == 'source':
                continue
        os.makedirs(val, exist_ok=True)

    total_time = segment_tiling(**directories, patch_size=patch_size, mag_level=mag, step_size=step_size, index=index)
    print(f"The average processing time for each slide is {total_time:.2f} seconds.")


@click.command()
@click.option('--source_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing slide images. Used only when --source_list is not provided. Supports .svs, .ndpi, and .scn files.')
@click.option('--source_list', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Text file with one slide path per line. Mutually exclusive with --source_dir.')
@click.option('--save_dir', type=click.Path(file_okay=False, dir_okay=True), required=True, help='path to the save directory')
@click.option('--patch_size', type=int, default=256, show_default=True, help='patch size')
@click.option('--step_size', type=int, default=256, show_default=True, help='step size')
@click.option('--mag', type=int, default=20, show_default=True, help='magnification for patch extraction')
@click.option('--index', type=int, default=0, show_default=True,
              help='Batch index used only to name slide_info_<index>.csv when tiling slides in multiple jobs.')
def generate_tiles(source_dir, source_list, save_dir, patch_size, step_size, mag, index):
    if bool(source_dir) == bool(source_list):
        raise click.UsageError("Provide exactly one of --source_dir or --source_list.")
    tile_slide_images(source_dir, source_list, save_dir, patch_size, step_size, mag, index)


def main():
    generate_tiles()


if __name__ == '__main__':
    main()
