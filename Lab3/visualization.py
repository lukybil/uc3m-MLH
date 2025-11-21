import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
from scipy.spatial import ConvexHull
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA


def denormalize_mnist(img):
    """Denormalize MNIST images from [-1, 1] to [0, 1]"""
    return (img * 0.5 + 0.5).clamp(0, 1)


def denormalize_svhn(img):
    """Denormalize SVHN images from [-1, 1] to [0, 1]"""
    return (img * 0.5 + 0.5).clamp(0, 1)


def save_reconstruction_samples(model, test_loader, config, num_samples=10):
    """
    Save reconstruction samples comparing original and reconstructed images.
    Collects one sample per digit (0-9) and displays in vertical A4-friendly layout.
    """
    model.eval()

    # Collect one sample per digit class
    digit_samples = {i: None for i in range(10)}

    with torch.no_grad():
        for mnist_imgs, svhn_imgs, labels in test_loader:
            for idx, label in enumerate(labels):
                label_int = label.item()
                if digit_samples[label_int] is None:
                    digit_samples[label_int] = {
                        "mnist": mnist_imgs[idx : idx + 1].to(config.device),
                        "svhn": svhn_imgs[idx : idx + 1].to(config.device),
                        "label": labels[idx : idx + 1].to(config.device),
                    }

            # Check if we have all digits
            if all(v is not None for v in digit_samples.values()):
                break

    # Reconstruct each digit
    mnist_imgs_list = []
    svhn_imgs_list = []
    mnist_recon_list = []
    svhn_recon_list = []

    with torch.no_grad():
        for digit in range(10):
            sample = digit_samples[digit]
            if model.use_label_conditioning:
                output = model(sample["mnist"], sample["svhn"], sample["label"])
            else:
                output = model(sample["mnist"], sample["svhn"])

            mnist_imgs_list.append(sample["mnist"].cpu())
            svhn_imgs_list.append(sample["svhn"].cpu())
            mnist_recon_list.append(output["mnist_recon"].cpu())
            svhn_recon_list.append(output["svhn_recon"].cpu())

    # Stack all samples
    mnist_imgs = torch.cat(mnist_imgs_list, dim=0)
    svhn_imgs = torch.cat(svhn_imgs_list, dim=0)
    mnist_recon = torch.cat(mnist_recon_list, dim=0)
    svhn_recon = torch.cat(svhn_recon_list, dim=0)

    # Denormalize images
    mnist_imgs = denormalize_mnist(mnist_imgs)
    mnist_recon = denormalize_mnist(mnist_recon)
    svhn_imgs = denormalize_svhn(svhn_imgs)
    svhn_recon = denormalize_svhn(svhn_recon)

    # Plot MNIST reconstruction
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axes[0, i].imshow(mnist_imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        axes[1, i].imshow(mnist_recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.suptitle("MNIST Reconstruction")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "mnist_reconstruction.png"), dpi=150)
    plt.close()

    # Plot SVHN reconstruction
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axes[0, i].imshow(svhn_imgs[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        axes[1, i].imshow(svhn_recon[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.suptitle("SVHN Reconstruction")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "svhn_reconstruction.png"), dpi=150)
    plt.close()

    print(f"Reconstruction samples saved to {config.results_dir}")


def save_generation_samples(model, config, num_samples=10):
    """
    Generate and save samples from the latent space.
    If label conditioning is enabled, generates one sample per digit (0-9).
    """
    model.eval()

    with torch.no_grad():
        if model.use_label_conditioning:
            # Generate one sample for each digit class (0-9)
            labels = torch.arange(10).to(config.device)
            output = model.generate(num_samples=10, device=config.device, labels=labels)
        else:
            output = model.generate(num_samples, config.device)

        mnist_gen = output["mnist_gen"].cpu()
        svhn_gen = output["svhn_gen"].cpu()

    # Denormalize generated images
    mnist_gen = denormalize_mnist(mnist_gen)
    svhn_gen = denormalize_svhn(svhn_gen)

    # Plot generated MNIST
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(mnist_gen[i].squeeze(), cmap="gray")
        axes[i].axis("off")
        if model.use_label_conditioning:
            axes[i].set_title(f"Digit {i}", fontsize=10)

    plt.suptitle("Generated MNIST")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "mnist_generation.png"), dpi=150)
    plt.close()

    # Plot generated SVHN
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(svhn_gen[i].permute(1, 2, 0))
        axes[i].axis("off")
        if model.use_label_conditioning:
            axes[i].set_title(f"Digit {i}", fontsize=10)

    plt.suptitle("Generated SVHN")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "svhn_generation.png"), dpi=150)
    plt.close()

    print(f"Generation samples saved to {config.results_dir}")


def save_cross_generation_samples(model, test_loader, config, num_samples=10):
    """
    Generate and save cross-domain generation samples.
    """
    model.eval()

    # Get a batch of test data
    mnist_imgs, svhn_imgs, labels = next(iter(test_loader))
    mnist_imgs = mnist_imgs[:num_samples].to(config.device)
    svhn_imgs = svhn_imgs[:num_samples].to(config.device)
    labels = labels[:num_samples].to(config.device)

    with torch.no_grad():
        # MNIST to SVHN (use labels if available)
        if model.use_label_conditioning:
            mnist_to_svhn = model.cross_generate(
                mnist_imgs, source_type="mnist", labels=labels
            )
            svhn_gen_from_mnist = mnist_to_svhn["mnist_to_svhn"].cpu()

            # SVHN to MNIST
            svhn_to_mnist = model.cross_generate(
                svhn_imgs, source_type="svhn", labels=labels
            )
            mnist_gen_from_svhn = svhn_to_mnist["svhn_to_mnist"].cpu()
        else:
            mnist_to_svhn = model.cross_generate(mnist_imgs, source_type="mnist")
            svhn_gen_from_mnist = mnist_to_svhn["mnist_to_svhn"].cpu()

            # SVHN to MNIST
            svhn_to_mnist = model.cross_generate(svhn_imgs, source_type="svhn")
            mnist_gen_from_svhn = svhn_to_mnist["svhn_to_mnist"].cpu()

    mnist_imgs = mnist_imgs.cpu()
    svhn_imgs = svhn_imgs.cpu()

    # Denormalize images
    mnist_imgs = denormalize_mnist(mnist_imgs)
    svhn_imgs = denormalize_svhn(svhn_imgs)
    svhn_gen_from_mnist = denormalize_svhn(svhn_gen_from_mnist)
    mnist_gen_from_svhn = denormalize_mnist(mnist_gen_from_svhn)

    # Plot MNIST to SVHN
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axes[0, i].imshow(mnist_imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Source MNIST", fontsize=10)

        axes[1, i].imshow(svhn_gen_from_mnist[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Generated SVHN", fontsize=10)

    plt.suptitle("Cross-Generation: MNIST → SVHN")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "mnist_to_svhn.png"), dpi=150)
    plt.close()

    print(f"Cross-generation samples (all digits) saved to {config.results_dir}")

    # Plot SVHN to MNIST
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axes[0, i].imshow(svhn_imgs[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Source SVHN", fontsize=10)

        axes[1, i].imshow(mnist_gen_from_svhn[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Generated MNIST", fontsize=10)

    plt.suptitle("Cross-Generation: SVHN → MNIST")
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "svhn_to_mnist.png"), dpi=150)
    plt.close()

    print(f"Cross-generation samples saved to {config.results_dir}")


def save_comprehensive_visualization(model, test_loader, config):
    """
    Save a comprehensive visualization with all tasks on one A4 page:
    - Reconstruction (10 digits in rows)
    - Generation (10 digits in rows)
    - Cross-generation (10 digits in rows)
    """
    model.eval()

    # Collect one sample per digit class
    digit_samples = {i: None for i in range(10)}

    for mnist_imgs, svhn_imgs, labels in test_loader:
        for idx, label in enumerate(labels):
            label_int = label.item()
            if digit_samples[label_int] is None:
                digit_samples[label_int] = {
                    "mnist": mnist_imgs[idx : idx + 1].to(config.device),
                    "svhn": svhn_imgs[idx : idx + 1].to(config.device),
                    "label": labels[idx : idx + 1].to(config.device),
                }

        if all(v is not None for v in digit_samples.values()):
            break

    # 1. RECONSTRUCTION
    mnist_imgs_list = []
    svhn_imgs_list = []
    mnist_recon_list = []
    svhn_recon_list = []

    with torch.no_grad():
        for digit in range(10):
            sample = digit_samples[digit]
            if model.use_label_conditioning:
                output = model(sample["mnist"], sample["svhn"], sample["label"])
            else:
                output = model(sample["mnist"], sample["svhn"])

            mnist_imgs_list.append(sample["mnist"].cpu())
            svhn_imgs_list.append(sample["svhn"].cpu())
            mnist_recon_list.append(output["mnist_recon"].cpu())
            svhn_recon_list.append(output["svhn_recon"].cpu())

    mnist_imgs = torch.cat(mnist_imgs_list, dim=0)
    svhn_imgs = torch.cat(svhn_imgs_list, dim=0)
    mnist_recon = torch.cat(mnist_recon_list, dim=0)
    svhn_recon = torch.cat(svhn_recon_list, dim=0)

    # 2. GENERATION
    with torch.no_grad():
        if model.use_label_conditioning:
            labels = torch.arange(10).to(config.device)
            output = model.generate(num_samples=10, device=config.device, labels=labels)
        else:
            output = model.generate(10, config.device)

        mnist_gen = output["mnist_gen"].cpu()
        svhn_gen = output["svhn_gen"].cpu()

    # 3. CROSS-GENERATION
    svhn_gen_from_mnist_list = []
    mnist_gen_from_svhn_list = []

    with torch.no_grad():
        for digit in range(10):
            sample = digit_samples[digit]

            if model.use_label_conditioning:
                mnist_to_svhn = model.cross_generate(
                    sample["mnist"], source_type="mnist", labels=sample["label"]
                )
                svhn_to_mnist = model.cross_generate(
                    sample["svhn"], source_type="svhn", labels=sample["label"]
                )
            else:
                mnist_to_svhn = model.cross_generate(
                    sample["mnist"], source_type="mnist"
                )
                svhn_to_mnist = model.cross_generate(sample["svhn"], source_type="svhn")

            svhn_gen_from_mnist_list.append(mnist_to_svhn["mnist_to_svhn"].cpu())
            mnist_gen_from_svhn_list.append(svhn_to_mnist["svhn_to_mnist"].cpu())

    svhn_gen_from_mnist = torch.cat(svhn_gen_from_mnist_list, dim=0)
    mnist_gen_from_svhn = torch.cat(mnist_gen_from_svhn_list, dim=0)

    # Denormalize all images
    mnist_imgs = denormalize_mnist(mnist_imgs)
    mnist_recon = denormalize_mnist(mnist_recon)
    svhn_imgs = denormalize_svhn(svhn_imgs)
    svhn_recon = denormalize_svhn(svhn_recon)
    mnist_gen = denormalize_mnist(mnist_gen)
    svhn_gen = denormalize_svhn(svhn_gen)
    svhn_gen_from_mnist = denormalize_svhn(svhn_gen_from_mnist)
    mnist_gen_from_svhn = denormalize_mnist(mnist_gen_from_svhn)

    # Create comprehensive A4 layout
    # Layout: 10 rows x 10 columns
    # Columns: [MNIST Orig | MNIST Recon | SVHN Orig | SVHN Recon | Gen MNIST | Gen SVHN | Src MNIST | M→S | Src SVHN | S→M]
    fig, axes = plt.subplots(10, 10, figsize=(8.27, 10))  # A4 portrait

    for digit in range(10):
        # Reconstruction: MNIST
        axes[digit, 0].imshow(mnist_imgs[digit].squeeze(), cmap="gray")
        axes[digit, 0].axis("off")
        if digit == 0:
            axes[digit, 0].set_title("M\nOrig", fontsize=7)
        axes[digit, 0].text(
            -0.15,
            0.5,
            f"{digit}",
            transform=axes[digit, 0].transAxes,
            fontsize=8,
            va="center",
            ha="right",
            fontweight="bold",
        )

        axes[digit, 1].imshow(mnist_recon[digit].squeeze(), cmap="gray")
        axes[digit, 1].axis("off")
        if digit == 0:
            axes[digit, 1].set_title("M\nRecon", fontsize=7)

        axes[digit, 2].imshow(svhn_imgs[digit].permute(1, 2, 0))
        axes[digit, 2].axis("off")
        if digit == 0:
            axes[digit, 2].set_title("S\nOrig", fontsize=7)

        axes[digit, 3].imshow(svhn_recon[digit].permute(1, 2, 0))
        axes[digit, 3].axis("off")
        if digit == 0:
            axes[digit, 3].set_title("S\nRecon", fontsize=7)

        # Generation
        axes[digit, 4].imshow(mnist_gen[digit].squeeze(), cmap="gray")
        axes[digit, 4].axis("off")
        if digit == 0:
            axes[digit, 4].set_title("Gen\nM", fontsize=7)

        axes[digit, 5].imshow(svhn_gen[digit].permute(1, 2, 0))
        axes[digit, 5].axis("off")
        if digit == 0:
            axes[digit, 5].set_title("Gen\nS", fontsize=7)

        # Cross-generation
        axes[digit, 6].imshow(mnist_imgs[digit].squeeze(), cmap="gray")
        axes[digit, 6].axis("off")
        if digit == 0:
            axes[digit, 6].set_title("Src\nM", fontsize=7)

        axes[digit, 7].imshow(svhn_gen_from_mnist[digit].permute(1, 2, 0))
        axes[digit, 7].axis("off")
        if digit == 0:
            axes[digit, 7].set_title("M→\nS", fontsize=7)

        axes[digit, 8].imshow(svhn_imgs[digit].permute(1, 2, 0))
        axes[digit, 8].axis("off")
        if digit == 0:
            axes[digit, 8].set_title("Src\nS", fontsize=7)

        axes[digit, 9].imshow(mnist_gen_from_svhn[digit].squeeze(), cmap="gray")
        axes[digit, 9].axis("off")
        if digit == 0:
            axes[digit, 9].set_title("S→\nM", fontsize=7)

    # Add section dividers with text
    fig.text(0.195, 0.965, "Reconstruction", ha="center", fontsize=9, fontweight="bold")
    fig.text(0.485, 0.965, "Generation", ha="center", fontsize=9, fontweight="bold")
    fig.text(
        0.755, 0.965, "Cross-Generation", ha="center", fontsize=9, fontweight="bold"
    )

    plt.suptitle(
        "Multi-View VAE: All Tasks (Digits 0-9)",
        fontsize=11,
        y=0.995,
        fontweight="bold",
    )
    plt.subplots_adjust(
        left=0.02, right=1.0, top=0.93, bottom=0.01, hspace=0.01, wspace=0.05
    )
    plt.savefig(
        os.path.join(config.results_dir, "comprehensive_visualization.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Comprehensive visualization saved to {config.results_dir}")


def visualize_latent_space(model, test_loader, config, num_samples=1000):
    """
    Visualize the latent space using t-SNE.
    Samples approximately equal numbers from each digit class.
    """

    model.eval()

    class_samples = {i: {"latent": [], "labels": []} for i in range(10)}
    samples_per_class = num_samples // 10

    with torch.no_grad():
        for mnist_imgs, svhn_imgs, labels in test_loader:
            mnist_imgs = mnist_imgs.to(config.device)
            svhn_imgs = svhn_imgs.to(config.device)
            labels = labels.to(config.device)

            if model.use_label_conditioning:
                output = model(mnist_imgs, svhn_imgs, labels)
            else:
                output = model(mnist_imgs, svhn_imgs)

            z = output["mu"].cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(z)):
                label = labels_np[i]
                if len(class_samples[label]["latent"]) < samples_per_class:
                    class_samples[label]["latent"].append(z[i])
                    class_samples[label]["labels"].append(label)

            if all(
                len(class_samples[i]["latent"]) >= samples_per_class for i in range(10)
            ):
                break

    latent_codes = []
    labels_list = []
    for class_id in range(10):
        latent_codes.extend(class_samples[class_id]["latent"])
        labels_list.extend(class_samples[class_id]["labels"])

    latent_codes = np.array(latent_codes)
    labels_array = np.array(labels_list)

    unique_labels = np.unique(labels_array)
    print(
        f"Collected {len(latent_codes)} samples with {len(unique_labels)} unique digit classes: {sorted(unique_labels)}"
    )
    print(f"Samples per class: {[len(class_samples[i]['latent']) for i in range(10)]}")

    print(f"Latent space statistics:")
    print(f"  Mean: {latent_codes.mean():.4f}")
    print(f"  Std: {latent_codes.std():.4f}")
    print(f"  Min: {latent_codes.min():.4f}")
    print(f"  Max: {latent_codes.max():.4f}")

    print("Applying t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(latent_codes) // 10),
        learning_rate="auto",
        early_exaggeration=12.0,
        random_state=42,
        init="pca",
    )
    latent_2d = tsne.fit_transform(latent_codes)

    plt.figure(figsize=(10, 8))

    cmap = plt.cm.tab10

    for digit_class in range(10):
        class_mask = labels_array == digit_class
        class_points = latent_2d[class_mask]

        if len(class_points) >= 3:
            centroid = class_points.mean(axis=0)
            distances = np.linalg.norm(class_points - centroid, axis=1)
            threshold = np.percentile(distances, 90)
            inlier_mask = distances <= threshold
            inliers = class_points[inlier_mask]

            if len(inliers) >= 3:
                try:
                    hull = ConvexHull(inliers)
                    hull_points = inliers[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])

                    color = cmap(digit_class)
                except:
                    pass

    scatter = plt.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=labels_array,
        cmap="tab10",
        alpha=0.7,
        s=20,
        vmin=0,
        vmax=9,
        edgecolors="black",
        linewidth=0.3,
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit Class", rotation=270, labelpad=20)
    plt.title("t-SNE Visualization by Digit Class")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "latent_space_tsne.png"), dpi=150)
    plt.close()

    # Compute clustering quality metrics
    silhouette = silhouette_score(latent_2d, labels_array)
    davies_bouldin = davies_bouldin_score(latent_2d, labels_array)
    calinski_harabasz = calinski_harabasz_score(latent_2d, labels_array)

    # Also compute metrics on original high-dimensional latent codes
    silhouette_hd = silhouette_score(latent_codes, labels_array)
    davies_bouldin_hd = davies_bouldin_score(latent_codes, labels_array)
    calinski_harabasz_hd = calinski_harabasz_score(latent_codes, labels_array)

    print(f"\nClustering metrics (t-SNE 2D):")
    print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")

    print(f"\nClustering metrics (Original {latent_codes.shape[1]}D latent space):")
    print(f"  Silhouette Score: {silhouette_hd:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin_hd:.4f}")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz_hd:.2f}")

    print(f"\nt-SNE visualization saved to {config.results_dir}")


def visualize_latent_space_pca(model, test_loader, config, num_samples=1000):
    """
    Alternative visualization using PCA (faster and deterministic).
    """

    model.eval()

    # Collect samples (same as before)
    class_samples = {i: {"latent": [], "labels": []} for i in range(10)}
    samples_per_class = num_samples // 10

    with torch.no_grad():
        for mnist_imgs, svhn_imgs, labels in test_loader:
            mnist_imgs = mnist_imgs.to(config.device)
            svhn_imgs = svhn_imgs.to(config.device)
            labels = labels.to(config.device)

            if model.use_label_conditioning:
                output = model(mnist_imgs, svhn_imgs, labels)
            else:
                output = model(mnist_imgs, svhn_imgs)

            z = output["mu"].cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(z)):
                label = labels_np[i]
                if len(class_samples[label]["latent"]) < samples_per_class:
                    class_samples[label]["latent"].append(z[i])
                    class_samples[label]["labels"].append(label)

            if all(
                len(class_samples[i]["latent"]) >= samples_per_class for i in range(10)
            ):
                break

    latent_codes = []
    labels_list = []
    for class_id in range(10):
        latent_codes.extend(class_samples[class_id]["latent"])
        labels_list.extend(class_samples[class_id]["labels"])

    latent_codes = np.array(latent_codes)
    labels_array = np.array(labels_list)

    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=labels_array,
        cmap="tab10",
        alpha=0.7,
        s=20,
        vmin=0,
        vmax=9,
        edgecolors="black",
        linewidth=0.3,
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit Class", rotation=270, labelpad=20)

    plt.title(
        f"PCA Visualization of Latent Space\n(Explained variance: {sum(pca.explained_variance_ratio_):.2%})"
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "latent_space_pca.png"), dpi=150)
    plt.close()

    print(f"PCA visualization saved to {config.results_dir}")


def plot_training_history(history, config):
    """
    Plot training history curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, history["train_recon_loss"], label="Train")
    axes[1].plot(epochs, history["test_recon_loss"], label="Test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(epochs, history["train_kl_div"], label="Train")
    axes[2].plot(epochs, history["test_kl_div"], label="Test")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL Divergence")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "training_history.png"), dpi=150)
    plt.close()

    print(f"Training history plot saved to {config.results_dir}")
