import os
import glob
import argparse
from importlib.resources import files
from pathlib import Path


def default_pipeline_config_path():
    return str(files("hetairos").joinpath("pipeline_config.yaml"))


def default_model_config_path():
    return str(files("hetairos.aggregator_train_val").joinpath("config.yaml"))


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_pipeline_config(config_path):
    import yaml

    default_path = Path(default_pipeline_config_path())
    with open(default_path) as f:
        config = yaml.safe_load(f) or {}

    requested_path = Path(config_path)
    if requested_path.resolve() != default_path.resolve():
        with open(requested_path) as f:
            overrides = yaml.safe_load(f) or {}
        config = deep_update(config, overrides)

    return config


def resolve_model_config_path(model_config, pipeline_config_path):
    if model_config is None:
        return default_model_config_path()
    model_config_path = Path(model_config).expanduser()
    if not model_config_path.is_absolute():
        model_config_path = Path(pipeline_config_path).expanduser().resolve().parent / model_config_path
    return str(model_config_path)


def validate_arguments(parser, args):
    if args.tiling:
        if bool(args.slide_dir) == bool(args.slide_list):
            parser.error("Provide exactly one of --slide_dir or --slide_list when --tiling is used")
    if args.feature_extraction:
        if args.tile_paths_file and args.tile_dir:
            parser.error("Provide only one of --tile_paths_file or --tile_dir")
        if args.tile_paths_file is None and args.tile_dir is None and not args.tiling:
            parser.error("--tile_paths_file or --tile_dir is required when --feature_extraction is used without --tiling")
    if args.model_run:
        if args.dataset is None and not args.feature_extraction:
            parser.error("--dataset is required when --model_run is used without --feature_extraction")
        if args.mode == 'train' and args.split is None:
            parser.error("--split is required when --model_run --mode train is used")
        if args.label is None:
            parser.error("--label is required when --model_run is used")
        if args.label_map is None:
            parser.error("--label_map is required when --model_run is used")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='End-to-end Hetairos pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tiling', action='store_true', help='Run WSI tiling')
    parser.add_argument('--feature_extraction', action='store_true', help='Run tile feature extraction')
    parser.add_argument('--model_run', action='store_true', help='Run model training or testing')

    tiling_group = parser.add_argument_group('Tiling arguments')
    tiling_group.add_argument('--slide_dir', type=str, help='Directory containing slide images. Mutually exclusive with --slide_list. Supports .svs, .ndpi, and .scn files.')
    tiling_group.add_argument('--slide_list', type=str, help='Text file with one slide path per line. Mutually exclusive with --slide_dir.')
    tiling_group.add_argument('--tile_savedir', type=str, default='./tiles/', help='Directory where tiles, masks, stitches, and slide_info are saved')

    feature_group = parser.add_argument_group('Feature extraction arguments')
    feature_group.add_argument('--tile_paths_file', type=str, default=None,
                        help='Text file with one tile-image directory per line. If omitted after --tiling, generated from <tile_savedir>/tiles/* slide directories.')
    feature_group.add_argument('--tile_dir', type=str, default=None,
                        help='Path to a directory containing tile images directly, or a root directory with one slide tile directory per subfolder. Mutually exclusive with --tile_paths_file.')
    feature_group.add_argument('--feature_dir', type=str, default='./features/', help='Directory where h5_files and pt_files are saved')

    model_group = parser.add_argument_group('Model run arguments')
    model_group.add_argument('--dataset', type=str, default=None,
                        help='Path to the dataset directory containing .pt feature files. If omitted after --feature_extraction, uses <feature_dir>/pt_files.')
    model_group.add_argument('--label', type=str, default=None,
                        help='Path to the slide label CSV file, which should contain columns including slide, family, probability vector, age, and location')
    model_group.add_argument('--label_map', type=str, default=None, help='Path to label mapping file')
    model_group.add_argument('--split', type=str, default=None,
                        help='Path to the dataset split file (YAML) containing train and test slide IDs, structured as {"train": [slide_id], "test": [slide_id]}. Required for train mode; if omitted in test mode, generated from dataset .pt files.')
    model_group.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Operation mode')
    model_group.add_argument('--exp_name', type=str, default='default_exp', help='Identifier for the experiment')
    model_group.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save predictions')
    parser.add_argument('--config', type=str, default=default_pipeline_config_path(),
                        help='Pipeline YAML for less frequently changed settings such as tiling size, feature batch size, model hyperparameters, and trainer hardware.')

    args = parser.parse_args()
    if not (args.tiling or args.feature_extraction or args.model_run):
        parser.error("At least one of --tiling, --feature_extraction, or --model_run must be provided")
    validate_arguments(parser, args)
    return args


def main():
    # Run the pipeline
    args = parse_arguments()
    pipeline_cfg = load_pipeline_config(args.config)
    tiling_cfg = pipeline_cfg.get('tiling', {})
    feature_cfg = pipeline_cfg.get('feature_extraction', {})
    run_cfg = pipeline_cfg.get('model_run', {})

    if args.tiling:
        from hetairos.preprocessing.tiling.main_create_tiles import tile_slide_images

        tile_slide_images(
            source_dir=args.slide_dir,
            source_list=args.slide_list,
            save_dir=args.tile_savedir,
            patch_size=tiling_cfg.get('patch_size', 256),
            step_size=tiling_cfg.get('step_size', 256),
            mag=tiling_cfg.get('mag', 20),
            index=tiling_cfg.get('index', 0),
        )

    if args.feature_extraction:
        from hetairos.preprocessing.feature_extraction.get_features import extract_features

        if args.tile_paths_file is None and args.tile_dir is None:
            try:
                tile_paths_file = Path(args.tile_savedir) / 'tile_paths.txt'
                tile_dirs = [
                    item for item in sorted(glob.glob(os.path.join(args.tile_savedir, 'tiles', '*')))
                    if os.path.isdir(item)
                ]
                if not tile_dirs:
                    raise RuntimeError(f"No tile directories found under {os.path.join(args.tile_savedir, 'tiles')}")
                with open(tile_paths_file, 'w') as f:
                    for item in tile_dirs:
                        f.write("%s\n" % item)
                args.tile_paths_file = str(tile_paths_file)
            except OSError as exc:
                raise RuntimeError('No tile folder found, please provide --tile_paths_file or --tile_dir') from exc

        extract_features(
            tile_paths_file=args.tile_paths_file,
            tile_dir=args.tile_dir,
            batchsize=feature_cfg.get('batchsize', 768),
            feature_dir=args.feature_dir,
        )

    if args.model_run:
        import yaml
        from hetairos.aggregator_train_val.model_run import model_run
        from hetairos.aggregator_train_val.utils import read_yaml

        if args.dataset is None:
            args.dataset = os.path.join(args.feature_dir, 'pt_files')
        if args.split is None:
            if args.mode == 'train':
                raise ValueError("--split is required when --mode train is used")
            testset = os.listdir(args.dataset)
            testset = [os.path.splitext(item)[0] for item in testset if item.endswith('.pt')]
            split_path = os.path.join(os.path.dirname(os.path.abspath(args.dataset)), 'split.yaml')
            with open(split_path, 'w') as f:
                yaml.dump({'train': [], 'test': testset}, f)
            args.split = split_path

        model_config_path = resolve_model_config_path(run_cfg.get('model_config'), args.config)
        cfg = read_yaml(model_config_path)
        cfg['Data']['data_dir'] = args.dataset
        cfg['Data']['data_split'] = args.split
        cfg['Data']['label_file'] = args.label
        cfg['Data']['soft_labels'] = run_cfg.get('soft_labels', False)
        cfg['Data']['aug'] = run_cfg.get('data_aug', False)
        cfg['Data']['label_file'] = args.label
        cfg['Data']['label_mapping'] = args.label_map
        cfg['General']['mode'] = args.mode
        cfg['General']['accelerator'] = run_cfg.get('accelerator', cfg['General'].get('accelerator', 'auto'))
        cfg['General']['devices'] = run_cfg.get('devices', cfg['General'].get('devices', 'auto'))
        cfg['General']['precision'] = run_cfg.get('precision', cfg['General'].get('precision', '32-true'))
        cfg['Model']['exp_name'] = args.exp_name
        cfg['Model']['name'] = run_cfg.get('model', cfg['Model'].get('name', 'ATransMIL'))
        cfg['Model']['group_num'] = run_cfg.get('groups', cfg['Model'].get('group_num', 3))
        cfg['Model']['n_classes'] = run_cfg.get('classes', cfg['Model'].get('n_classes', 186))
        cfg['Model']['cl_w'] = run_cfg.get('cl_weight', cfg['Model'].get('cl_w', 20))
        cfg['Data']['preds_save'] = args.output_dir
        cfg['resume'] = run_cfg.get('resume', False)
        model_run(cfg)


if __name__ == '__main__':
    main()
