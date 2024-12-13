import os
import glob
import yaml
import argparse
from preprocessing.tiling.main_create_tiles import tile_slide_images
from preprocessing.feature_extraction.get_features import extract_features
from aggregator_train_val.model_run import model_run
from aggregator_train_val.utils import read_yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description='End to end pipeline')
    parser.add_argument('--tiling', action='store_true', help='Whether to perform tiling')
    parser.add_argument('--feature_extraction', action='store_true', help='Whether to perform feature extraction')
    parser.add_argument('--model_run', action='store_true', help='Whether to perform training/testing')
    args, unknown = parser.parse_known_args()
    if not (args.tiling or args.feature_extraction or args.model_run):
        parser.error("At least one of --tiling, --feature_extraction, or --model_run must be True")
    
    if args.tiling:
        tiling_group = parser.add_argument_group('Tiling arguments')
        tiling_group.add_argument('--slide_dir', type=str, help='path to the source slide image (.svs) directory')
        tiling_group.add_argument('--slide_list', type=str, help='path to the source slide image list (.txt) to be processed')
        tiling_group.add_argument('--tile_savedir', type=str, default='./tiles/', help='path to the save directory')
    
    if args.feature_extraction:
        feature_group = parser.add_argument_group('Feature extraction arguments')
        feature_group.add_argument('--tile_dir', type=str, default=None, 
                            help='path to the tile folder (.txt). If not provided, the tile folder will be generated from the tiling results')
        feature_group.add_argument('--batchsize', type=int, default=768, help='batch size for inference')
        feature_group.add_argument('--feature_dir', type=str, default='./features/', help='path to the save directory')
        
    if args.model_run:
        model_group = parser.add_argument_group('Model run arguments')
        model_group.add_argument('--dataset', type=str, default=None, 
                            help='Path to the dataset directory. If not provided, the dataset will be generated from the feature extraction results')
        model_group.add_argument('--label', type=str, default='./aggregator_train_val/labels/labels.csv',
                            help='Path to the slide label CSV file, which should contain columns including slide, family, probability vector, age, and location')
        model_group.add_argument('--label_map', type=str, default='./aggregator_train_val/annot_files/class_ID.yaml', help='Path to label mapping file')
        model_group.add_argument('--split', type=str, default=None,
                            help='Path to the dataset split file (YAML) containing train and test slide IDs, structured as {"train": [slide_id], "test": [slide_id]}. If not provided, the file will be generated from the dataset')
        model_group.add_argument('--mode', type=str, default='train', help='Operation mode: train or test') 
        model_group.add_argument('--exp_name', type=str, default='default_exp', help='Identifier for the experiment') 
        model_group.add_argument('--output_dir', type=str, default='./aggregator_train_val/predictions', help='Directory to save predictions') 
        model_group.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint') 
        model_group.add_argument('--config', type=str, default='./aggregator_train_val/config.yaml', help='Path to configuration file') 
        model_group.add_argument('--data_aug', action='store_true', help='Apply data augmentation during training')
        model_group.add_argument('--soft_labels', action='store_true', help='Use soft labels during training') 

    return parser.parse_args() 
    
    
if __name__ == '__main__':
    # Run the pipeline
    args = parse_arguments()
    
    if args.tiling:
        tile_slide_images(source_dir=args.slide_dir, source_list=args.slide_list, save_dir=args.tile_savedir)
    
    if args.feature_extraction:
        if args.tile_dir is None:
            try:
                with open('./tile_list.txt', 'w') as f:
                    for item in glob.glob(os.path.join(args.tile_savedir, 'tiles','*')):
                        f.write("%s\n" % item)
                args.tile_dir = './tile_list.txt'
            except:
                print('No tile folder found, please provide the tile folder path')
                exit()
                
        extract_features(split=args.tile_dir, batchsize=args.batchsize, feature_dir=args.feature_dir)
        
    if args.model_run:
        if args.dataset is None:
            try:
                args.dataset = os.path.join(args.feature_dir, 'pt_files')
            except:
                print('No dataset found, please provide the dataset path')
                exit()
        if args.split is None:
            try:
                testset = os.listdir(args.dataset)
                testset = [os.path.splitext(item)[0] for item in testset]
                with open('./aggregator_train_val/split.yaml', 'w') as f:
                    yaml.dump({'train': [], 'test': testset}, f)
                args.split = './aggregator_train_val/split.yaml'
            except:
                print('No split file found, please provide the split file path')
                exit()
        
        cfg = read_yaml(args.config)
        cfg['Data']['data_dir'] = args.dataset
        cfg['Data']['data_split'] = args.split
        cfg['Data']['label_file'] = args.label
        cfg['Data']['soft_labels'] = args.soft_labels
        cfg['Data']['aug'] = args.data_aug
        cfg['Data']['label_file'] = args.label
        cfg['Data']['label_mapping'] = args.label_map
        cfg['General']['mode'] = args.mode
        cfg['Model']['exp_name'] = args.exp_name
        cfg['Model']['preds_save'] = args.output_dir
        cfg['resume'] = args.resume
        model_run(cfg)