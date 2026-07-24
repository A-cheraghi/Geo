from asyncio.log import logger
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.save_helper import load_checkpoint

parser = argparse.ArgumentParser(description='Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # setup_model_and_freeze(model, cfg['trainer'], device, logger)



    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()










def setup_model_and_freeze(model, cfg_trainer, device, logger):
    # 1. Loading pretrain model (فقط اگر pretrain فعال باشد)
    if cfg_trainer.get('pretrain_model'):
        assert os.path.exists(cfg_trainer['pretrain_model'])
        load_checkpoint(model=model,
                        optimizer=None,
                        filename=cfg_trainer['pretrain_model'],
                        map_location=device,
                        logger=logger)

    # 2. Freeze the whole pretrained network
    for param in model.parameters():
        param.requires_grad = False

    # 3. Enable training only for the correction network
    train_modules = [
        model.fusion_mlp,
        model.box_correction,
        model.dim_correction,
        model.depth_correction,
        model.angle_correction,
        model.class_correction
    ]
    for module in train_modules:
        for param in module.parameters():
            param.requires_grad = True

    # 4. Initialize correction heads with zero output only for the first training run
    # (اگر حالت resume باشد این بخش اجرا نمی‌شود تا وزن‌های آموزش‌دیده پاک نشوند)
    if cfg_trainer.get('pretrain_model') and not cfg_trainer.get('resume_model'):
        correction_heads = [
            model.box_correction,
            model.dim_correction,
            model.depth_correction,
            model.angle_correction,
            model.class_correction
        ]
        for head in correction_heads:
            linear_layers = [
                m for m in head.modules()
                if isinstance(m, nn.Linear)
            ]
            last_layer = linear_layers[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)     
        logger.info("Correction heads last layers weights and biases initialized to zero.")






if __name__ == '__main__':
    main()
