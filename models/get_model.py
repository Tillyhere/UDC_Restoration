"""
Get model
"""
from models.guided_filter import DeepAtrousGuidedFilter
from models.lr_net import SmoothDilatedResidualAtrousGuidedBlock, LRNet
from models.our_lr_net import OurLRNet
from models.lp_lr_net_high3_E import LPLRNet_E
from models.lp_lr_net_high3_B import LPLRNet_B
from models.lp_lr_net_high3_G import LPLRNet_G
from models.lp_lr_net_high3_H import LPLRNet_H
from models.lp_lr_net_high3_K import LPLRNet_K
from models.lp_lr_net_high3_Q import LPLRNet_Q
from models.lp_lr_net_high3_T import LPLRNet_T
from models.lp_lr_net_high3_V import LPLRNet_V
from models.lp_lr_net_high3_X import LPLRNet_X
from models.lp_lr_net_high3_II import LPLRNet_II
from models.lp_lr_net_high3_VII import LPLRNet_VII
from models.lp_lr_net_high3_VII_re5 import LPLRNet_VII_re5
from models.lp_lr_net_high3_XVII import LPLRNet_XVII
from models.lp_lr_net_high2_XIX import LPLRNet_XIX
from models.lp_lr_net_high3_XX import LPLRNet_XX
from models.lp_lr_net_high3_XXI import LPLRNet_XXI
from models.lp_lr_net_high3_VII_allres import LPLRNet_VII_allres
from models.lp_lr_net_high3_VII_resize import LPLRNet_VII_resize


def our_model(args):
    return OurLRNet(args)


def model(args):
    return DeepAtrousGuidedFilter(args)

def lplr_E(args):
    return LPLRNet_E(args)

def lplr_B(args):
    return LPLRNet_B(args)

def lplr_G(args):
    return LPLRNet_G(args)

def lplr_H(args):
    return LPLRNet_H(args)

def lplr_K(args):
    return LPLRNet_K(args)

def lplr_Q(args):
    return LPLRNet_Q(args)

def lplr_T(args):
    return LPLRNet_T(args)

def lplr_V(args):
    return LPLRNet_V(args)

def lplr_X(args):
    return LPLRNet_X(args)

def lplr_II(args):
    return LPLRNet_II(args)

def lplr_VII(args):
    return LPLRNet_VII(args)

def lplr_VII_re5(args):
    return LPLRNet_VII_re5(args)

def lplr_XVII(args):
    return LPLRNet_XVII(args)

def lplr_XIX(args):
    return LPLRNet_XIX(args)

def lplr_XX(args):
    return LPLRNet_XX(args)

def lplr_XXI(args):
    return LPLRNet_XXI(args)

def lplr_VII_allres(args):
    return LPLRNet_VII_allres(args)

def lplr_VII_resize(args):
    return LPLRNet_VII_resize(args)
