import argparse
import os
import sys

# os.chdir(sys.path[0])
# sys.path.append('../')
# os.getcwd()

from .badnet import BadNet, add_common_attack_args
# from backdoorbench.attack.badnet import BadNet, add_common_attack_args


class Blended(BadNet):
    '''
    this script is for blended attack
    
    @article{Blended,
    	title        = {Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning},
    	author       = {Xinyun Chen and Chang Liu and Bo Li and Kimberly Lu and Dawn Song},
    	journal      = {arXiv preprint arXiv:1712.05526},
    	year         = {2017}
    }
    '''

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--attack_trigger_img_path", type=str, )
        parser.add_argument("--attack_train_blended_alpha", type=float, )
        parser.add_argument("--attack_test_blended_alpha", type=float, )
        parser.add_argument('--bd_yaml_path', type=str, default=os.path.abspath(os.path.dirname(__file__))+"/"+'../config/attack/blended/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser


if __name__ == '__main__':
    attack = Blended()
    attack.attack()
