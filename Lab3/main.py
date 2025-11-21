import torch
import os
import argparse
from config import Config
from dataset import load_datasets
from multiview_vae import MultiViewVAE
from trainer import train_model
from visualization import (
    save_reconstruction_samples,
    save_generation_samples,
    save_cross_generation_samples,
    visualize_latent_space,
    plot_training_history,
)
from debug_utils import (
    verify_label_mapping,
    visualize_paired_samples,
    analyze_cross_generation_by_digit,
    print_model_summary,
)


def main(args):
    config = Config()
    if args.expert_type:
        config.expert_type = args.expert_type
    if args.epochs:
        config.num_epochs = args.epochs
    if args.latent_dim:
        config.latent_dim = args.latent_dim
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    print("=" * 80)
    print("Multi-View VAE for MNIST and SVHN")
    print("=" * 80)

    # load datasets
    print("\nLoading datasets...")
    train_loader, test_loader = load_datasets(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # verify label mapping
    verify_label_mapping(train_loader, test_loader, config)
    visualize_paired_samples(train_loader, config, num_samples=5)

    print("\nInitializing model...")
    model = MultiViewVAE(config).to(config.device)
    print_model_summary(model)

    if args.mode == "train":
        print("\nStarting training...")
        model, history = train_model(model, train_loader, test_loader, config)

        print("\nPlotting training history...")
        plot_training_history(history, config)

        final_model_path = os.path.join(config.checkpoint_dir, "final_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "history": history,
            },
            final_model_path,
        )
        print(f"Final model saved to {final_model_path}")

    elif args.mode == "evaluate":
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(config.checkpoint_dir, "final_model.pt")

        print(f"\nLoading model from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=config.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully")

    # visualize results
    if args.mode in ["train", "evaluate"]:
        print("\nGenerating visualizations...")

        print("Reconstruction samples...")
        save_reconstruction_samples(
            model, test_loader, config, num_samples=config.num_samples
        )

        print("Generation from latent space...")
        save_generation_samples(model, config, num_samples=config.num_samples)

        print("Cross-domain generation...")
        save_cross_generation_samples(
            model, test_loader, config, num_samples=config.num_samples
        )

        print("Latent space visualization (t-SNE)...")
        visualize_latent_space(model, test_loader, config)

        print("Cross-generation analysis by digit...")
        analyze_cross_generation_by_digit(model, test_loader, config)

        print(f"\nResults saved to {config.results_dir}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-View VAE for MNIST and SVHN")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Mode: train or evaluate",
    )
    parser.add_argument(
        "--expert_type",
        type=str,
        default=None,
        choices=["poe", "moe"],
        help="Expert combination type (poe or moe)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=None, help="Latent space dimensionality"
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation"
    )

    args = parser.parse_args()
    main(args)
