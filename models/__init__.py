from .p2pnet import build
from .p2pnet_confi import build_confi
from .p2pnet_confi_128_single import build_confi_128_single
from .p2pnet_confi_32 import build_confi_32
from .p2pnet_confi_16 import build_confi_16
from .p2pnet_confi_8 import build_confi_8
from .p2pnet_confi_4 import build_confi_4

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    return build(args, training)

def build_model_confi(args, training = False):
    return build_confi(args, training)

def build_model_128_single(args, training = False):
    return build_confi_128_single(args, training)

def build_model_32(args, training = False):
    return build_confi_32(args, training)

def build_model_16(args, training = False):
    return build_confi_16(args, training)
    
def build_model_8(args, training = False):
    return build_confi_8(args, training)

def build_model_4(args, training = False):
    return build_confi_4(args, training)