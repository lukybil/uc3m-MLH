import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def verify_label_mapping(train_loader, test_loader, config):
    """
    Verify that labels are correctly mapped between MNIST and SVHN.
    """
    print("\n" + "=" * 80)
    print("LABEL MAPPING VERIFICATION")
    print("=" * 80)

    # Check training set
    train_labels = []
    for _, _, labels in train_loader:
        train_labels.extend(labels.numpy())

    # Check test set
    test_labels = []
    for _, _, labels in test_loader:
        test_labels.extend(labels.numpy())

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    print("\nTraining set label distribution:")
    for digit in range(10):
        count = np.sum(train_labels == digit)
        print(f"  Digit {digit}: {count} samples ({count/len(train_labels)*100:.2f}%)")

    print("\nTest set label distribution:")
    for digit in range(10):
        count = np.sum(test_labels == digit)
        print(f"  Digit {digit}: {count} samples ({count/len(test_labels)*100:.2f}%)")

    # Verify all digits 0-9 are present
    train_unique = set(train_labels)
    test_unique = set(test_labels)

    if train_unique == set(range(10)):
        print("\n✓ All digits 0-9 present in training set")
    else:
        print(f"\n✗ Missing digits in training set: {set(range(10)) - train_unique}")

    if test_unique == set(range(10)):
        print("✓ All digits 0-9 present in test set")
    else:
        print(f"✗ Missing digits in test set: {set(range(10)) - test_unique}")


def visualize_paired_samples(train_loader, config, num_samples=5):
    """
    Visualize paired MNIST-SVHN samples to verify correct pairing.
    """
    print("\n" + "=" * 80)
    print("PAIRED SAMPLES VISUALIZATION")
    print("=" * 80)

    mnist_imgs, svhn_imgs, labels = next(iter(train_loader))

    # Denormalize images from [-1, 1] to [0, 1]
    mnist_imgs = (mnist_imgs * 0.5 + 0.5).clamp(0, 1)
    svhn_imgs = (svhn_imgs * 0.5 + 0.5).clamp(0, 1)

    # Select samples for each digit
    fig, axes = plt.subplots(10, 2 * num_samples, figsize=(2 * num_samples * 2, 20))

    for digit in range(10):
        digit_indices = (labels == digit).nonzero(as_tuple=True)[0][:num_samples]

        if len(digit_indices) == 0:
            print(f"Warning: No samples found for digit {digit}")
            continue

        for idx, sample_idx in enumerate(digit_indices):
            # MNIST
            axes[digit, idx * 2].imshow(mnist_imgs[sample_idx].squeeze(), cmap="gray")
            axes[digit, idx * 2].axis("off")
            if idx == 0:
                axes[digit, idx * 2].set_title(f"Digit {digit}\nMNIST", fontsize=8)

            # SVHN
            axes[digit, idx * 2 + 1].imshow(svhn_imgs[sample_idx].permute(1, 2, 0))
            axes[digit, idx * 2 + 1].axis("off")
            if idx == 0:
                axes[digit, idx * 2 + 1].set_title(f"Digit {digit}\nSVHN", fontsize=8)

    plt.suptitle("Paired MNIST-SVHN Samples by Digit", fontsize=16)
    plt.tight_layout()
    os.makedirs(config.results_dir, exist_ok=True)
    plt.savefig(
        os.path.join(config.results_dir, "paired_samples_verification.png"), dpi=150
    )
    plt.close()

    print(
        f"Paired samples visualization saved to {config.results_dir}/paired_samples_verification.png"
    )


def analyze_cross_generation_by_digit(model, test_loader, config):
    """
    Analyze cross-generation performance for each digit separately.
    """
    print("\n" + "=" * 80)
    print("CROSS-GENERATION ANALYSIS BY DIGIT")
    print("=" * 80)

    model.eval()

    # Collect samples for each digit
    digit_samples = {i: {"mnist": [], "svhn": [], "labels": []} for i in range(10)}

    with torch.no_grad():
        for mnist_imgs, svhn_imgs, labels in test_loader:
            mnist_imgs = mnist_imgs.to(config.device)
            svhn_imgs = svhn_imgs.to(config.device)
            labels_np = labels.numpy()

            for digit in range(10):
                digit_mask = labels == digit
                if digit_mask.sum() == 0:
                    continue

                digit_mnist = mnist_imgs[digit_mask][:5]  # Take up to 5 samples
                digit_svhn = svhn_imgs[digit_mask][:5]
                digit_labels = labels[digit_mask][:5]

                if len(digit_mnist) > 0:
                    digit_samples[digit]["mnist"].append(digit_mnist)
                    digit_samples[digit]["svhn"].append(digit_svhn)
                    digit_samples[digit]["labels"].append(digit_labels)

    # Generate cross-domain samples for each digit
    fig, axes = plt.subplots(10, 6, figsize=(18, 30))

    for digit in range(10):
        if len(digit_samples[digit]["mnist"]) == 0:
            continue

        mnist_batch = torch.cat(digit_samples[digit]["mnist"])[:3]
        svhn_batch = torch.cat(digit_samples[digit]["svhn"])[:3]
        labels_batch = torch.cat(digit_samples[digit]["labels"])[:3]

        # SVHN to MNIST
        with torch.no_grad():
            if model.use_label_conditioning:
                svhn_to_mnist_output = model.cross_generate(
                    svhn_batch, source_type="svhn", labels=labels_batch
                )
                mnist_to_svhn_output = model.cross_generate(
                    mnist_batch, source_type="mnist", labels=labels_batch
                )
            else:
                svhn_to_mnist_output = model.cross_generate(
                    svhn_batch, source_type="svhn"
                )
                mnist_to_svhn_output = model.cross_generate(
                    mnist_batch, source_type="mnist"
                )

            mnist_from_svhn = svhn_to_mnist_output["svhn_to_mnist"].cpu()
            svhn_from_mnist = mnist_to_svhn_output["mnist_to_svhn"].cpu()

        # Denormalize images from [-1, 1] to [0, 1]
        mnist_batch_denorm = (mnist_batch.cpu() * 0.5 + 0.5).clamp(0, 1)
        svhn_batch_denorm = (svhn_batch.cpu() * 0.5 + 0.5).clamp(0, 1)
        mnist_from_svhn_denorm = (mnist_from_svhn * 0.5 + 0.5).clamp(0, 1)
        svhn_from_mnist_denorm = (svhn_from_mnist * 0.5 + 0.5).clamp(0, 1)

        # Plot: Original SVHN | Generated MNIST | Original MNIST
        sample_idx = 0
        axes[digit, 0].imshow(svhn_batch_denorm[sample_idx].permute(1, 2, 0))
        axes[digit, 0].set_title(f"Digit {digit}\nSVHN", fontsize=8)
        axes[digit, 0].axis("off")

        axes[digit, 1].imshow(mnist_from_svhn_denorm[sample_idx].squeeze(), cmap="gray")
        axes[digit, 1].set_title(f"SVHN→MNIST", fontsize=8)
        axes[digit, 1].axis("off")

        axes[digit, 2].imshow(mnist_batch_denorm[sample_idx].squeeze(), cmap="gray")
        axes[digit, 2].set_title(f"Real MNIST", fontsize=8)
        axes[digit, 2].axis("off")

        # Plot: Original MNIST | Generated SVHN | Original SVHN
        axes[digit, 3].imshow(mnist_batch_denorm[sample_idx].squeeze(), cmap="gray")
        axes[digit, 3].set_title(f"MNIST", fontsize=8)
        axes[digit, 3].axis("off")

        axes[digit, 4].imshow(svhn_from_mnist_denorm[sample_idx].permute(1, 2, 0))
        axes[digit, 4].set_title(f"MNIST→SVHN", fontsize=8)
        axes[digit, 4].axis("off")

        axes[digit, 5].imshow(svhn_batch_denorm[sample_idx].permute(1, 2, 0))
        axes[digit, 5].set_title(f"Real SVHN", fontsize=8)
        axes[digit, 5].axis("off")

    plt.suptitle("Cross-Generation Analysis by Digit", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.results_dir, "cross_generation_by_digit.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Cross-generation analysis saved to {config.results_dir}/cross_generation_by_digit.png"
    )


def print_model_summary(model):
    """
    Print detailed model architecture summary.
    """
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Latent dimension: {model.latent_dim}")
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Expert type: {model.expert_type}")
    print(f"Label conditioning: {model.use_label_conditioning}")

    if model.use_label_conditioning:
        print(f"Number of classes: {model.num_classes}")

    print("\nComponent parameters:")
    print(
        f"  MNIST Encoder: {sum(p.numel() for p in model.mnist_encoder.parameters()):,}"
    )
    print(
        f"  MNIST Decoder: {sum(p.numel() for p in model.mnist_decoder.parameters()):,}"
    )
    print(
        f"  SVHN Encoder: {sum(p.numel() for p in model.svhn_encoder.parameters()):,}"
    )
    print(
        f"  SVHN Decoder: {sum(p.numel() for p in model.svhn_decoder.parameters()):,}"
    )
