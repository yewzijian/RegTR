import os, argparse

from easydict import EasyDict

from cvhelpers.misc import prepare_logger

from data_loaders import get_dataloader
from models import get_model
from trainer import Trainer
from utils.misc import load_config


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, help='Benchmark dataset', default='3DMatch',
                    choices=['3DMatch', '3DLoMatch', 'ModelNet', 'ModelLoNet'])
# General
parser.add_argument('--config', type=str, help='Path to the config file.')
# Logging
parser.add_argument('--logdir', type=str, default='../logs',
                    help='Directory to store logs, summaries, checkpoints.')
parser.add_argument('--dev', action='store_true',
                    help='If true, will ignore logdir and log to ../logdev instead')
parser.add_argument('--name', type=str,
                    help='Prefix to add to logging directory')
# Misc
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker threads for dataloader')
# Training and model options
parser.add_argument('--resume', type=str, help='Checkpoint to resume from')


opt = parser.parse_args()
logger, opt.log_path = prepare_logger(opt)
# Override config if --resume is passed
if opt.config is None:
    if opt.resume is None or not os.path.exists(opt.resume):
        logger.error('--config needs to be supplied unless resuming from checkpoint')
        exit(-1)
    else:
        resume_folder = opt.resume if os.path.isdir(opt.resume) else os.path.dirname(opt.resume)
        opt.config = os.path.normpath(os.path.join(resume_folder, '../config.yaml'))
        if os.path.exists(opt.config):
            logger.info(f'Using config file from checkpoint directory: {opt.config}')
        else:
            logger.error('Config not found in resume directory')
            exit(-2)
else:
    # Save config to log
    config_out_fname = os.path.join(opt.log_path, 'config.yaml')
    with open(opt.config, 'r') as in_fid, open(config_out_fname, 'w') as out_fid:
        out_fid.write(f'# Original file name: {opt.config}\n')
        out_fid.write(in_fid.read())
cfg = EasyDict(load_config(opt.config))


def main():

    if cfg.dataset == '3dmatch':
        assert opt.benchmark in ['3DMatch', '3DLoMatch'], \
            "Benchmark for 3dmatch dataset must be one of ['3DMatch', '3DLoMatch']"
        cfg.benchmark = opt.benchmark
    elif cfg.dataset == 'modelnet':
        assert opt.benchmark in ['ModelNet', 'ModelLoNet'], \
            "Benchmark for modelnet dataset must be one of ['ModelNet', 'ModelLoNet']"
        cfg.partial = [0.7, 0.7] if opt.benchmark == 'ModelNet' else [0.5, 0.5]

    test_loader = get_dataloader(cfg, phase='test')
    Model = get_model(cfg.model)
    model = Model(cfg)
    trainer = Trainer(opt, niter=cfg.niter, grad_clip=cfg.grad_clip)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
