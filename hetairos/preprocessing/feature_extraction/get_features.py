import os
import h5py
import click
import glob
import torch
import timm
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


# imagenet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)

class roi_dataset(Dataset):
    def __init__(self, tile_dirs):
        super().__init__()
        self.tile_dirs = tile_dirs
        self.tile_ls = []
        for tile_dir in self.tile_dirs:
            self.tile_ls.extend(sorted(glob.glob(os.path.join(tile_dir, '*.jpg'))))
        if len(self.tile_ls) == 0:
            raise ValueError("No .jpg tile images found in the provided tile directories.")
        self.transform = trnsfrms_val

    def __len__(self):
        return len(self.tile_ls)

    def __getitem__(self, idx):
        slide_id = os.path.basename(os.path.dirname(self.tile_ls[idx]))
        image = Image.open(self.tile_ls[idx]).convert('RGB')
        image = self.transform(image)
        spatial_x = int(os.path.basename(self.tile_ls[idx]).split('_')[-2])
        spatial_y = int(os.path.splitext(os.path.basename(self.tile_ls[idx]).split('_')[-1])[0])
        return image, slide_id, spatial_x, spatial_y


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='w'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def find_tile_dirs(tile_dir):
    direct_tiles = sorted(glob.glob(os.path.join(tile_dir, '*.jpg')))
    if direct_tiles:
        return [tile_dir]

    tile_dirs = []
    for item in sorted(glob.glob(os.path.join(tile_dir, '*'))):
        if os.path.isdir(item) and glob.glob(os.path.join(item, '*.jpg')):
            tile_dirs.append(item)
    return tile_dirs


def resolve_tile_dirs(tile_paths_file=None, tile_dir=None):
    if bool(tile_paths_file) == bool(tile_dir):
        raise ValueError("Provide exactly one of tile_paths_file or tile_dir.")
    if tile_paths_file:
        with open(tile_paths_file) as f:
            tile_dirs = [line.strip() for line in f if line.strip()]
    else:
        tile_dirs = find_tile_dirs(tile_dir)
        if not tile_dirs:
            raise ValueError(f"No .jpg tile images found in {tile_dir} or its immediate subdirectories.")
    missing_dirs = [path for path in tile_dirs if not os.path.isdir(path)]
    if missing_dirs:
        raise FileNotFoundError(f"Tile directories not found: {missing_dirs}")
    return tile_dirs


def extract_features(tile_paths_file=None, batchsize=768, feature_dir='./features', tile_dir=None):
    tile_dirs = resolve_tile_dirs(tile_paths_file=tile_paths_file, tile_dir=tile_dir)
    test_datat = roi_dataset(tile_dirs)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=batchsize, shuffle=False)

    # change the name to the model you want to use here
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    # Bug fix A: device-adaptive instead of hardcoded .cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print('Inference begins...')
    with torch.no_grad():
        for batch, slide_id, spatial_x, spatial_y in tqdm(database_loader, total=len(database_loader), desc='Extracting features', unit='batch'):
            batch = batch.to(device)  # Bug fix A: device-adaptive

            features = model(batch)
            features = features.cpu().numpy()
            id_set = list(np.unique(np.array(slide_id)))
            spatial_x = np.array(spatial_x)
            spatial_y = np.array(spatial_y)
            for id in id_set:
                feature = features[np.array(slide_id)==id]
                pos_x = spatial_x[np.array(slide_id)==id]
                pos_y = spatial_y[np.array(slide_id)==id]
                output_path = os.path.join(feature_dir, 'h5_files', id+'.h5')
                asset_dict = {'features': feature, 'pos_x': pos_x, 'pos_y': pos_y}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode='a')

    h5_ls = [os.path.join(feature_dir, 'h5_files', os.path.basename(os.path.normpath(item))) for item in tile_dirs]
    os.makedirs(os.path.join(feature_dir, 'pt_files'), exist_ok=True)
    for idx, h5file in enumerate(h5_ls):
        if os.path.exists(os.path.join(feature_dir, 'pt_files', os.path.basename(h5file)+'.pt')):
            pass
        else:
            file = h5py.File(h5file+'.h5', "r")
            features = file['features'][:]
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(feature_dir, 'pt_files', os.path.basename(h5file)+'.pt'))
            file.close()

    print('Feature extraction done!')


@click.command()
@click.option('--tile_paths_file', '--split', 'tile_paths_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Text file with one tile-image directory per line. Each directory should contain tile JPGs for one slide, named like <slide_id>_x_y_<x>_<y>.jpg. Legacy alias: --split.')
@click.option('--tile_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing tile JPGs directly, or a root directory containing one subdirectory per slide. Mutually exclusive with --tile_paths_file.')
@click.option('--batchsize', type=int, default=768, show_default=True, help='batch size for inference')
@click.option('--feature_dir', type=click.Path(file_okay=False, dir_okay=True), default='./features', show_default=True, help='path to the save directory')
def inference(tile_paths_file, tile_dir, batchsize, feature_dir):
    if bool(tile_paths_file) == bool(tile_dir):
        raise click.UsageError("Provide exactly one of --tile_paths_file/--split or --tile_dir.")
    extract_features(tile_paths_file=tile_paths_file, tile_dir=tile_dir, batchsize=batchsize, feature_dir=feature_dir)


def main():
    inference()


if __name__ == '__main__':
    main()
