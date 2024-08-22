from segmentation.tools import train as train_mmseg
from segmentation.tools.model_converters import dinovit2mmseg
import argparse
import os


def main(args):

    # Convert the model if needed
    if not os.path.exists(str(os.path.join(args.output_dir, 'converted.pth'))):
        print('Converting the checkpoint to MMSEG format...')

        converter_args = {}
        converter_args['src'] = str(os.path.join(args.output_dir, 'checkpoint.pth'))
        converter_args['dst'] = str(os.path.join(args.output_dir, 'converted.pth'))
        dinovit2mmseg.main(argparse.Namespace(**converter_args))
        print('... model successfully converted.')

    # Get the default args from mmmseg
    mmseg_args = vars(train_mmseg.parse_args())
    extra_args = vars(args)

    # Update the args
    mmseg_args['config'] = os.path.join('segmentation/configs', extra_args['mmseg_config_name'])
    work_dir = f"{extra_args['mmseg_work_dir']}_{extra_args['mmseg_frozen_stages']}_{extra_args['mmseg_lr']}_{extra_args['mmseg_ntokens']}"
    mmseg_args['work_dir'] = str(os.path.join(args.output_dir, work_dir))

    # Merge the extra args
    extra_args['backbone_weights'] = str(os.path.join(args.output_dir, 'converted.pth'))

    # Launch
    train_mmseg.main(argparse.Namespace(**mmseg_args), extra_args)


def get_parser():
    parser = argparse.ArgumentParser("Launcher for the segmentation evaluations")
    parser.add_argument("--output_dir", default="dir/containing/the/checkpoint.pth", type=str)
    parser.add_argument("--mmseg_config_name", default="segmenter/name_of_the_config.py", type=str)
    parser.add_argument("--mmseg_work_dir", default="", type=str)
    parser.add_argument("--mmseg_frozen_stages", default=12, type=int)
    parser.add_argument("--mmseg_samples_per_gpu", default=4, type=int)
    parser.add_argument("--mmseg_lr", default=0.01, type=float)
    parser.add_argument("--mmseg_wd", default=0.01, type=float)
    parser.add_argument("--mmseg_workers_per_gpu", default=12, type=int)
    parser.add_argument("--mmseg_ntokens", default=1, type=int)
    return parser

    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)