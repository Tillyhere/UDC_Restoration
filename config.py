from pathlib import Path
import torch

from utils.self_ensemble import ensemble_ops


def base_config():
    exp_name = "ours"

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    image_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    val_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    output_dir = Path("outputs") / exp_name
    ckpt_dir = Path("ckpts") / exp_name
    run_dir = Path("runs") / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    train_source_dir = image_dir / "Poled_train" / "LQ"
    train_target_dir = image_dir / "Poled_train" / "HQ"
    
    val_source_dir = image_dir / "Poled_test" / "LQ"
    val_target_dir = image_dir / "Poled_test" / "HQ"
    
    val_mega_source_dir = val_dir/ "Poled_test" / "LQ"
    val_mega_target_dir = val_dir/"Poled_test" / "HQ"
    test_source_dir = image_dir / "Poled_test" / "LQ"

    static_val_image = "1.png"
    static_val_mega_image = "1.png"
    static_test_image = "1.png"

    image_height = 1024
    image_width = 2048
    do_resize = False
    h_size = 512
    w_size = 512
    batch_size = 4
    num_threads = batch_size  # parallel workers

    # augment
    do_augment = True

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    num_epochs=7000 

    learning_rate = 3e-4

    # Betas for AdamW. We follow https://arxiv.org/pdf/1704.00028
    beta_1 = 0.9
    beta_2 = 0.999

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    # saving models
    save_filename_G = "model.pth"
    save_filename_latest_G = "model_latest.pth"
    #save_filename_latest_G = "Epoch_61_model_latest.pth"
    # save a copy of weights every x epochs
    save_copy_every_epochs =1

    # For model ensembling
    save_num_snapshots = 1

    # the number of iterations (default: 10) to print at
    log_interval = 25

    # run val or test only every x epochs
    val_test_epoch_interval = 8 

    # ----------------------------------------------------------------------------  #
    # Val / Test Configs
    # ---------------------------------------------------------------------------- #

    # Self ensemble
    self_ensemble = False
    num_ensemble = len(ensemble_ops) + 1
    save_train = False

    inference_mode = "latest"
    assert inference_mode in ["latest", "best"]

    # ---------------------------------------------------------------------------- #
    # Model: See models/get_model.py for registry
    # ---------------------------------------------------------------------------- #
    pixelshuffle_ratio = 2

    # Guided map
    guided_map_kernel_size = 3
    guided_map_channels = 16

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_image = 1  # l1
    lambda_CoBi_RGB = 0.0  # https://arxiv.org/pdf/1905.05169.pdf

    cobi_rgb_patch_size = 8
    cobi_rgb_stride = 8

    resume = True
    finetune = False  # Wont load loss or epochs

    # ---------------------------------------------------------------------------- #
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    distdataparallel = False


def ours_poled():
    exp_name = "ours-poled"


def ours_poled_sim():
    exp_name = "ours-poled-sim"

    num_epochs = 16 + 32 + 64
    log_interval = 25
    val_test_epoch_interval = 3
    save_copy_every_epochs = 16

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    image_dir = Path("data")
    train_source_dir = image_dir / "Sim_train" / "POLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "POLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None


def ours_poled_PreTr():
    exp_name = "ours-poled-PreTr"


def ours_toled():
    exp_name = "ours-toled"

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def ours_toled_sim():
    exp_name = "ours-toled-sim"

    num_epochs = 16 + 32 + 64
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 16

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    image_dir = Path("data")
    train_source_dir = image_dir / "Sim_train" / "TOLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "TOLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None


def ours_toled_PreTr():
    exp_name = "ours-toled-PreTr"

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def ours_toled_patches():
    exp_name = "ours-toled-patches"


def ours_toled_patches_charbonnier():
    exp_name = "ours-toled-patches-charbonnier"
    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    
def ours_ori():
    exp_name = "ours-ori"


def poled_lp_lr_E():
    exp_name = 'poled-lp-lr-E'

def poled_lp_lr_B():
    exp_name = 'poled-lp-lr-B'

def poled_lp_lr_H():
    exp_name = 'poled-lp-lr-H'

def poled_lp_lr_G():
    exp_name = 'poled-lp-lr-G'

def poled_lp_lr_K():
    exp_name = 'poled-lp-lr-K'

def poled_lp_lr_Q():
    exp_name = 'poled-lp-lr-Q'

def poled_lp_lr_T():
    exp_name ='poled-lp-lr-T'

def poled_lp_lr_V():
    exp_name = 'poled-lp-lr-V'

def poled_lp_lr_X():
    exp_name = 'poled-lp-lr-X'

def poled_lp_lr_II():
    exp_name = 'poled-lp-lr-II'

def poled_lp_lr_VII():
    exp_name = 'poled-lp-lr-VII'

def poled_lp_lr_VII_re5():
    exp_name = 'poled-lp-lr-VII-re5'

def poled_lp_lr_VII_allres():
    exp_name = 'poled-lp-lr-VII-allres'

def poled_lp_lr_VII_resize():
    exp_name = 'poled-lp-lr-VII-resize'

def toled_lp_lr_XIX():
    exp_name = 'toled-lp-lr-XIX'
    image_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    val_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    output_dir = Path("outputs") / exp_name
    ckpt_dir = Path("ckpts") / exp_name
    run_dir = Path("runs") / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_test" / "LQ"
    val_target_dir = image_dir / "Toled_test" / "HQ"

    val_mega_source_dir = val_dir/ "Toled_test" / "LQ"
    val_mega_target_dir = val_dir/"Toled_test" / "HQ"
    test_source_dir = image_dir / "Toled_test" / "LQ"

def toled_lp_lr_XVII():
    exp_name = 'toled-lp-lr-XVII'
    image_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    val_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    output_dir = Path("outputs") / exp_name
    ckpt_dir = Path("ckpts") / exp_name
    run_dir = Path("runs") / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    val_mega_source_dir = val_dir/ "Toled_val" / "LQ"
    val_mega_target_dir = val_dir/"Toled_val" / "HQ"
    test_source_dir = image_dir / "Toled_test" / "LQ"


def toled_lp_lr_XX():
    exp_name = 'toled-lp-lr-XX'
    image_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    val_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    output_dir = Path("outputs") / exp_name
    ckpt_dir = Path("ckpts") / exp_name
    run_dir = Path("runs") / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_test" / "LQ"
    val_target_dir = image_dir / "Toled_test" / "HQ"

    val_mega_source_dir = val_dir/ "Toled_test" / "LQ"
    val_mega_target_dir = val_dir/"Toled_test" / "HQ"
    test_source_dir = image_dir / "Toled_test" / "LQ"

def toled_lp_lr_XXI():
    exp_name = 'toled-lp-lr-XXI'
    image_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    val_dir = Path("/home/ubuntu/tyhere/OLED_challenge")
    output_dir = Path("outputs") / exp_name
    ckpt_dir = Path("ckpts") / exp_name
    run_dir = Path("runs") / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_test" / "LQ"
    val_target_dir = image_dir / "Toled_test" / "HQ"

    val_mega_source_dir = val_dir/ "Toled_test" / "LQ"
    val_mega_target_dir = val_dir/"Toled_test" / "HQ"
    test_source_dir = image_dir / "Toled_test" / "LQ"


named_configs = [
    ours_poled,
    ours_poled_sim,
    ours_poled_PreTr,
    ours_toled,
    ours_toled_sim,
    ours_toled_PreTr,
    ours_toled_patches,
    ours_ori,
    ours_toled_patches_charbonnier,
    poled_lp_lr_E,
    poled_lp_lr_B,
    poled_lp_lr_H,
    poled_lp_lr_G,
    poled_lp_lr_K,
    poled_lp_lr_Q,
    poled_lp_lr_T,
    poled_lp_lr_V,
    poled_lp_lr_X,
    poled_lp_lr_II,
    poled_lp_lr_VII,
    poled_lp_lr_VII_re5,
    poled_lp_lr_VII_allres,
    poled_lp_lr_VII_resize,
    toled_lp_lr_XVII,
    toled_lp_lr_XIX,
    toled_lp_lr_XX,
    toled_lp_lr_XXI

]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
