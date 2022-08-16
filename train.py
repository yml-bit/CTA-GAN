#!/usr/bin/python3

import argparse
import os

from trainer import P2p_Trainer,Cyc_Trainer,Reg_Trainer,Hd_Trainer_x
import yaml
import warnings
import torch
import numpy as np
import random
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#nohup python -m visdom.server>>nop.out 2>&1&

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True / False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/HdGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    #CycleGan_x:Generator_xx
    #CycleGan:CycR_leGan
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)
    elif config['name'] == 'HdGan':#CTA-GAN  note:when trian,we shoud chage the name Hd_Trainer_x1/Hd_Trainer_x2 to Hd_Trainer_x
        trainer = Hd_Trainer_x(config)# hd1:add attention      hd2:doubule input and skip connnection
    # trainer.train()#a原始数据，原始输出。
    trainer.test()

###################################
if __name__ == '__main__':
    seed_everything(seed=42)
    main()