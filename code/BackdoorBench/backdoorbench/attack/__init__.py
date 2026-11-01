import os,sys
# sys.path.insert(0,os.path.dirname(__file__))
# print(sys.path)
from .badnet import BadNet
from .blended import Blended
from .blind import Blind
from .lf import LowFrequency
from .bpp import Bpp
from .inputaware import InputAware
from .lc import LabelConsistent
from .lira import LIRA
from .prototype import NormalCase 
from .sig import SIG
from .ssba import SSBA
from .trojannn import TrojanNN
from .wanet import Wanet

attack_classes = [
    'BadNet',
    'Blended',
    'Blind',
    'Bpp',
    'InputAware',
    'LabelConsistent',
    'LowFrequency',
    'LIRA',
    'NormalCase',
    'SIG',
    'SSBA',
    'TrojanNN',
    'Wanet'
]

__all__ = attack_classes
