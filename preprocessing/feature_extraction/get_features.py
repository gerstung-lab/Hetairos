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
    def __init__(self, slide_ls):
        super().__init__()
        self.slide_ls = slide_ls
        self.tile_ls = []
        for slide in self.slide_ls:
            self.tile_ls.extend(glob.glob(os.path.join(slide, '*.jpg')))
        self.transform = trnsfrms_val

    def __len__(self):
        return len(self.tile_ls)

    def __getitem__(self, idx):
        slide_id = self.tile_ls[idx].split('/')[-2]
        image = Image.open(self.tile_ls[idx]).convert('RGB')
        image = self.transform(image)
        spatial_x = int(self.tile_ls[idx].split('/')[-1].split('_')[-2])
        spatial_y = int(self.tile_ls[idx].split('/')[-1].split('_')[-1].split('.')[0])
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


def extract_features(split, batchsize=768, feature_dir='./features'):
    slide_ls = [line.rstrip('\n') for line in open(split)]
    os.remove(split) # remove the split file after reading
    test_datat=roi_dataset(slide_ls)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=batchsize, shuffle=False)
    
    # change the name to the model you want to use here
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)  
    
    model.cuda()
    model.eval()

    count = 0
    print('Inference begins...')
    with torch.no_grad():
        for batch, slide_id, spatial_x, spatial_y in database_loader:
            print(f'{count}/{len(database_loader)}')
            batch = batch.cuda()

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
            count += 1

    h5_ls = [os.path.join(feature_dir, 'h5_files', item.split('/')[-1]) for item in slide_ls]
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
@click.option('--split', type=str, help='path to the split file (.txt)')
@click.option('--batchsize', type=int, help='batch size for inference')
@click.option('--feature_dir', type=str, help='path to the save directory')
# @click.option('--ckpt', type=str, help='path to the save directory')
def inference(split, batchsize, feature_dir):
    extract_features(split, batchsize, feature_dir)


if __name__ == '__main__':
    inference()