import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import MultiPersonDataset
from unet import UNet
from config import get_config, list_available_models
import lpips

device = "cuda" if torch.cuda.is_available() else "mps"


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argument_parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset root directory containing all data (full_body_img, landmarks, aud_mel, identity_map.txt).",
    )
    argument_parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=list_available_models(),
        help="Model architecture size to use.",
    )
    argument_parser.add_argument("--save_dir", type=str, help="Training model save path.")
    argument_parser.add_argument("--epochs", type=int, default=300)
    argument_parser.add_argument(
        "--batchsize",
        type=int,
        default=None,
        help="Batch size (uses model default if not specified)",
    )
    argument_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (uses model default if not specified)",
    )
    argument_parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file path for starting training/fine-tuning.",
    )
    argument_parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of data loading worker threads."
    )
    argument_parser.add_argument(
        "--save_every_epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs.",
    )
    argument_parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Save checkpoint every N steps. If >0, uses this and disables epoch-based saving.",
    )

    argument_parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation.")

    return argument_parser.parse_args()


training_args = parse_arguments()


def execute_training(
    neural_network,
    total_epochs,
    batch_size,
    learning_rate,
    checkpoint_frequency,
):
    perceptual_loss_function = lpips.LPIPS(net="vgg").to(device)
    perceptual_loss_function.eval()

    checkpoint_directory = training_args.save_dir
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)

    optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)
    reconstruction_criterion = nn.L1Loss()

    starting_epoch = 0
    global_step_counter = 0
    if training_args.load_checkpoint and os.path.exists(training_args.load_checkpoint):
        neural_network.load_state_dict(torch.load(training_args.load_checkpoint, map_location=device))
        starting_epoch = 0
        global_step_counter = 0
    elif os.path.isdir(checkpoint_directory):
        existing_checkpoints = [
            f
            for f in os.listdir(checkpoint_directory)
            if f.startswith(f"{training_args.model_size}_") and f.endswith(".pth")
        ]
        if existing_checkpoints:

            def extract_step_or_epoch_number(filename):
                parts = filename.split("_")
                try:
                    if "step" in parts:
                        return int(parts[-1].split(".")[0])
                    return int(parts[-1].split(".")[0]) * 1000000
                except Exception:
                    return 0

            existing_checkpoints.sort(key=extract_step_or_epoch_number)
            most_recent_checkpoint = existing_checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_directory, most_recent_checkpoint)

            neural_network.load_state_dict(torch.load(checkpoint_path, map_location=device))

            filename_parts = most_recent_checkpoint.split("_")
            if "step" in filename_parts:
                try:
                    global_step_counter = int(filename_parts[-1].split(".")[0])
                except ValueError:
                    global_step_counter = 0
            else:
                try:
                    starting_epoch = int(filename_parts[-1].split(".")[0])
                except ValueError:
                    starting_epoch = 0

    training_dataset = MultiPersonDataset(
        training_args.dataset_dir,
        network_variant=training_args.model_size,
        enable_data_augmentation=not training_args.no_augmentation,
    )

    if not training_dataset:
        raise ValueError("dataset not found")

    data_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=training_args.num_workers,
        persistent_workers=True if training_args.num_workers > 0 else False,
    )

    if training_args.save_every_steps > 0:
        batches_per_epoch = len(data_loader)
        starting_epoch = global_step_counter // batches_per_epoch

    if starting_epoch == 0 and global_step_counter == 0:
        checkpoint_filename = (
            f"{training_args.model_size}_step_0.pth"
            if training_args.save_every_steps > 0
            else f"{training_args.model_size}_0.pth"
        )
        torch.save(neural_network.state_dict(), os.path.join(checkpoint_directory, checkpoint_filename))

    for epoch_number in range(starting_epoch, total_epochs):
        neural_network.train()

        with tqdm(total=len(training_dataset), desc=f"Epoch {epoch_number + 1}/{total_epochs}", unit="img") as progress_bar:
            for batch_data in data_loader:
                input_images, target_images, audio_features = batch_data
                input_images = input_images.to(device)
                target_images = target_images.to(device)
                audio_features = audio_features.to(device)

                predicted_images = neural_network(input_images, audio_features)

                perceptual_loss_value = perceptual_loss_function(
                    (predicted_images * 2.0 - 1.0), (target_images * 2.0 - 1.0)
                ).mean()
                perceptual_weight = 0.01
                perceptual_loss_name = "LPIPS"

                reconstruction_loss = reconstruction_criterion(predicted_images, target_images)
                total_loss = reconstruction_loss + perceptual_loss_value * perceptual_weight

                progress_bar.set_postfix(
                    **{
                        "L1": reconstruction_loss.item(),
                        perceptual_loss_name: perceptual_loss_value.item(),
                        "loss": total_loss.item(),
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                progress_bar.update(input_images.shape[0])
                global_step_counter += 1

                if (
                    training_args.save_every_steps > 0
                    and global_step_counter % training_args.save_every_steps == 0
                ):
                    checkpoint_filename = f"{training_args.model_size}_step_{global_step_counter}.pth"
                    torch.save(neural_network.state_dict(), os.path.join(checkpoint_directory, checkpoint_filename))

        if training_args.save_every_steps <= 0:
            if (epoch_number + 1) % checkpoint_frequency == 0 or epoch_number == total_epochs - 1:
                checkpoint_filename = f"{training_args.model_size}_{epoch_number + 1}.pth"
                torch.save(neural_network.state_dict(), os.path.join(checkpoint_directory, checkpoint_filename))


if __name__ == "__main__":
    network_configuration = get_config(training_args.model_size)

    batch_size = (
        training_args.batchsize if training_args.batchsize is not None else network_configuration.training_batch_size
    )
    learning_rate = training_args.lr if training_args.lr is not None else network_configuration.learning_rate

    neural_network = UNet(6, model_size=training_args.model_size).to(device)

    execute_training(
        neural_network,
        training_args.epochs,
        batch_size,
        learning_rate,
        training_args.save_every_epochs,
    )