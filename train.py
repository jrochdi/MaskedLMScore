import os
from datetime import datetime
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import UnitDataModule
from transformer_lightning_module import MaskedTransformerModule


def main():
    """
    Training entry point for masked HuBERT-unit Transformer LM.
    """
    config = load_config("config.yml")

    # Seeding & matmul precision
    pl.seed_everything(config["seed"])
    if "matmul_precision" in config:
        torch.set_float32_matmul_precision(config["matmul_precision"])

    # Generate unique run folder
    base_name = generate_base_name(config["training"]["log_name"])
    run_dir = os.path.join(config["training"]["runs_folder"], base_name)
    os.makedirs(run_dir, exist_ok=True)

    # Optionally save the config to the run folder
    with open(os.path.join(run_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    # Logging & callbacks (now run_dir is the root)
    logger = setup_logger(run_dir)
    callbacks = setup_callbacks(config, run_dir)

    # Data
    data_module = UnitDataModule(
        base_dir=config["data"]["clean_base_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        seq_len=config["data"]["seq_len"],
        pad_value=config["model"]["pad_id"],
        random_crop=config["data"]["random_crop"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        gibberish_base_dir=config["data"]["gibberish_base_dir"],  # <--- NEW
        pin_memory=True,
    )

    # Model
    model = MaskedTransformerModule(
        vocab_size=config["model"]["vocab_size"],
        pad_id=config["model"]["pad_id"],
        mask_id=config["model"]["mask_id"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["data"]["seq_len"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Trainer
    trainer = Trainer(
        logger=logger,
        accelerator=config["training"].get("accelerator", "gpu"),
        devices=config["training"].get("devices", -1),
        max_epochs=config["training"]["max_epochs"],
        min_epochs=config["training"]["min_epochs"],
        callbacks=callbacks,
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
    )

    trainer.fit(model, data_module)


def load_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def generate_base_name(log_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}-{log_name}"


def setup_logger(run_dir):
    """
    TensorBoard logger inside:
        <run_dir>/lightning_logs/
    """
    log_dir = os.path.join(run_dir, "lightning_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=run_dir,
        name="lightning_logs"
    )
    return logger


def setup_callbacks(config, run_dir):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_loss:.4e}",
        save_top_k=config["training"]["save_top_k"],
        monitor="val_loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["training"]["early_stopping_patience"],
        mode="min",
    )

    return [checkpoint_callback, early_stopping_callback]


if __name__ == "__main__":
    main()