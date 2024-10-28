import argparse, os, sys, datetime
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from utils.utils import instantiate_from_config
import warnings

warnings.filterwarnings("ignore")

workspace = Path("~/.workspace").expanduser().as_posix()
CONFIG_FILE_PATH = "configs/HDM_2D_Polyp.yaml"
# CONFIG_FILE_PATH = "configs/HDM_SUN-SEG.yaml"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    help="the path of the config file",
    default=CONFIG_FILE_PATH,
)
args = parser.parse_args()


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    sys.path.append(os.getcwd())

    config = OmegaConf.load(args.config_file)
    exp_config = config.pop("exp", OmegaConf.create())
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    logdir = Path(workspace).joinpath("logs", "HDM_exps", exp_config.exp_name, now)
    os.makedirs(logdir, exist_ok=True)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(exp_config.seed)

    trainer_opt = argparse.Namespace(**trainer_config)

    # model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": None,
                "save_dir": logdir,
                "version": "tensorboard",
            },
        },
    }
    if "logger" in lightning_config:
        logger_cfgs = lightning_config.logger
    else:
        logger_cfgs = OmegaConf.create()
    logger_cfgs = OmegaConf.merge(default_logger_cfgs, logger_cfgs)
    trainer_kwargs["logger"] = [instantiate_from_config(logger_cfgs[k]) for k in logger_cfgs]

    default_callbacks_cfg = {
        "checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "monitor": model.monitor,
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": False,
                "save_weights_only": True,
                "save_last": True,
                "save_top_k": 1,
            },
        },
        "setup_callback": {
            "target": "pl_utils.SetupCallback",
            "params": {
                "resume": exp_config.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "image_logger": {
            "target": "pl_utils.ImageLogger",
            "params": {"batch_frequency": 750, "max_images": 4, "clamp": True},
        },
        "learning_rate_logger": {
            "target": "pl_utils.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            },
        },
        "cuda_callback": {"target": "pl_utils.CUDACallback"},
    }

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if exp_config.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run
    if exp_config.train:
        trainer.fit(model, data)
