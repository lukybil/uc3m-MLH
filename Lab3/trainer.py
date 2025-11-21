import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import csv


def supervised_contrastive_loss(z, labels, temperature=0.07):
    """
    Supervised Contrastive Loss to improve latent space clustering.
    Pulls together embeddings from the same class and pushes apart different classes.

    Args:
        z: Latent codes [batch_size, latent_dim]
        labels: Class labels [batch_size]
        temperature: Temperature parameter for softmax

    Returns:
        Contrastive loss value
    """
    batch_size = z.size(0)

    z_normalized = F.normalize(z, p=2, dim=1)

    similarity_matrix = torch.matmul(z_normalized, z_normalized.t()) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(z.device)

    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask

    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    mask_sum = mask.sum(dim=1)

    mask_sum = torch.clamp(mask_sum, min=1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

    loss = -mean_log_prob_pos.mean()

    return loss


def adversarial_domain_loss(z_mnist, z_svhn):
    """
    Compute adversarial domain loss to encourage domain-invariant representations.

    Args:
        z_mnist: Latent codes from MNIST encoder
        z_svhn: Latent codes from SVHN encoder

    Returns:
        Domain loss (cross-entropy with flipped labels)
    """
    batch_size = z_mnist.size(0)

    # domain labels: 0 for MNIST, 1 for SVHN
    mnist_labels = torch.zeros(batch_size, dtype=torch.long, device=z_mnist.device)
    svhn_labels = torch.ones(batch_size, dtype=torch.long, device=z_svhn.device)

    z_combined = torch.cat([z_mnist, z_svhn], dim=0)
    labels_combined = torch.cat([mnist_labels, svhn_labels], dim=0)

    domain_logits = torch.stack(
        [
            -torch.mean(z_combined, dim=1),  # logit for class 0 (MNIST)
            torch.mean(z_combined, dim=1),  # logit for class 1 (SVHN)
        ],
        dim=1,
    )

    # cross-entropy loss with flipped labels to confuse the domain classifier
    domain_loss = F.cross_entropy(domain_logits, labels_combined, reduction="sum")

    return domain_loss


def mmd_loss(z_mnist, z_svhn, kernel="rbf", bandwidth=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) loss between MNIST and SVHN latent distributions.

    Args:
        z_mnist: Latent codes from MNIST encoder
        z_svhn: Latent codes from SVHN encoder
        kernel: Kernel type ('rbf' or 'linear')
        bandwidth: Bandwidth for RBF kernel

    Returns:
        MMD loss
    """

    def compute_kernel(x, y, kernel_type="rbf", bandwidth=1.0):
        """Compute kernel matrix."""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)

        if kernel_type == "rbf":
            # RBF kernel: exp(-||x-y||^2 / (2 * bandwidth^2))
            tiled_x = x.expand(x_size, y_size, dim)
            tiled_y = y.expand(x_size, y_size, dim)
            kernel_matrix = torch.exp(
                -torch.mean((tiled_x - tiled_y) ** 2, dim=2) / (2 * bandwidth)
            )
        else:  # linear kernel
            kernel_matrix = torch.mm(x.squeeze(1), y.squeeze(0).t())

        return kernel_matrix

    k_xx = compute_kernel(z_mnist, z_mnist, kernel, bandwidth)
    k_yy = compute_kernel(z_svhn, z_svhn, kernel, bandwidth)
    k_xy = compute_kernel(z_mnist, z_svhn, kernel, bandwidth)

    # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
    mmd_squared = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    return mmd_squared


def vae_loss(
    recon_mnist,
    recon_svhn,
    mnist_img,
    svhn_img,
    mu,
    logvar,
    beta=1.0,
    free_bits=0.0,
    classifier_logits=None,
    true_labels=None,
    z_mnist=None,
    z_svhn=None,
    z=None,
    gamma=0.0,
    domain_loss_type="mmd",
    svhn_recon_weight=1.0,
    contrastive_weight=5.0,
):
    """
    Returns:
        Total loss, reconstruction loss, KL divergence, classifier loss, domain loss, contrastive loss
    """
    # Reconstruction loss
    # MNIST: MSE for grayscale (1 channel)
    recon_loss_mnist = F.mse_loss(recon_mnist, mnist_img, reduction="sum")

    # SVHN: MSE for RGB (3 channels)
    # Apply configurable weight multiplier to balance RGB (3 channels) vs grayscale (1 channel)
    recon_loss_svhn = svhn_recon_weight * F.mse_loss(
        recon_svhn, svhn_img, reduction="sum"
    )

    recon_loss = recon_loss_mnist + recon_loss_svhn

    # KL divergence per dimension
    # KL per-dim: KL(q(z|x) || p(z)) = -1/2 * (1 + log(sigma^2) - mu^2 - sigma^2)
    # with logvar = log(sigma^2) so sigma^2 = exp(logvar)
    kl_div_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits > 0:
        kl_div_per_dim = torch.clamp(kl_div_per_dim, min=free_bits)

    kl_div = torch.sum(kl_div_per_dim)

    # auxiliary classifier loss to enforce label-informative latent codes
    clf_loss = 0.0
    if classifier_logits is not None and true_labels is not None:
        clf_loss = F.cross_entropy(classifier_logits, true_labels, reduction="sum")

    # domain loss for alignment with labels
    domain_loss = 0.0
    if gamma > 0 and z_mnist is not None and z_svhn is not None:
        if domain_loss_type == "adversarial":
            domain_loss = adversarial_domain_loss(z_mnist, z_svhn)
        elif domain_loss_type == "mmd":
            domain_loss = mmd_loss(z_mnist, z_svhn) * z_mnist.size(0)
        else:
            raise ValueError(f"Unknown domain loss type: {domain_loss_type}")

    # supervised contrastive loss to improve latent space clustering
    contrastive_loss = 0.0
    if z is not None and true_labels is not None and contrastive_weight > 0:
        contrastive_loss = supervised_contrastive_loss(z, true_labels) * z.size(0)

    total_loss = (
        recon_loss
        + beta * kl_div
        + 5.0 * clf_loss
        + gamma * domain_loss
        + contrastive_weight * contrastive_loss
    )

    return total_loss, recon_loss, kl_div, clf_loss, domain_loss, contrastive_loss


def train_epoch(model, train_loader, optimizer, config, epoch):
    """
    Returns:
        Average losses for the epoch
    """
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_kl_div = 0
    total_domain_loss = 0
    total_contrastive_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for mnist_img, svhn_img, labels in progress_bar:
        mnist_img = mnist_img.to(config.device)
        svhn_img = svhn_img.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()

        if model.use_label_conditioning:
            output = model(mnist_img, svhn_img, labels)
        else:
            output = model(mnist_img, svhn_img)

        if epoch <= config.beta_warmup_epochs:
            current_beta = config.beta_start + (config.beta_end - config.beta_start) * (
                epoch / config.beta_warmup_epochs
            )
        else:
            current_beta = config.beta_end

        gamma = getattr(config, "gamma", 0.0)
        domain_loss_type = getattr(config, "domain_loss_type", "mmd")
        contrastive_weight = getattr(config, "contrastive_weight", 5.0)

        loss, recon_loss, kl_div, clf_loss, domain_loss, contrastive_loss = vae_loss(
            output["mnist_recon"],
            output["svhn_recon"],
            mnist_img,
            svhn_img,
            output["mu"],
            output["logvar"],
            beta=current_beta,
            free_bits=config.free_bits,
            classifier_logits=output.get("classifier_logits"),
            true_labels=labels,
            z_mnist=output.get("z_mnist"),
            z_svhn=output.get("z_svhn"),
            z=output.get("z"),
            gamma=gamma,
            domain_loss_type=domain_loss_type,
            svhn_recon_weight=config.svhn_recon_weight,
            contrastive_weight=contrastive_weight,
        )

        loss.backward()
        optimizer.step()

        batch_size = mnist_img.size(0)
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_div += kl_div.item()
        total_domain_loss += (
            domain_loss if isinstance(domain_loss, float) else domain_loss.item()
        )
        total_contrastive_loss += (
            contrastive_loss
            if isinstance(contrastive_loss, float)
            else contrastive_loss.item()
        )
        num_batches += 1

        progress_bar.set_postfix(
            {
                "loss": loss.item() / batch_size,
                "recon": recon_loss.item() / batch_size,
                "kl": kl_div.item() / batch_size,
                "clf": (
                    clf_loss
                    if isinstance(clf_loss, float)
                    else clf_loss.item() / batch_size
                ),
                "domain": (
                    domain_loss
                    if isinstance(domain_loss, float)
                    else domain_loss.item() / batch_size
                ),
                "contr": (
                    contrastive_loss
                    if isinstance(contrastive_loss, float)
                    else contrastive_loss.item() / batch_size
                ),
            }
        )

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_kl_div = total_kl_div / len(train_loader.dataset)
    avg_domain_loss = total_domain_loss / len(train_loader.dataset)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader.dataset)

    return avg_loss, avg_recon_loss, avg_kl_div, avg_domain_loss, avg_contrastive_loss


def evaluate(model, test_loader, config):
    """
    Evaluate the model on test data.

    Args:
        model: MultiViewVAE model
        test_loader: Test data loader
        config: Configuration object

    Returns:
        Average losses on test set
    """
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_kl_div = 0
    total_domain_loss = 0
    total_contrastive_loss = 0

    with torch.no_grad():
        for mnist_img, svhn_img, labels in test_loader:
            mnist_img = mnist_img.to(config.device)
            svhn_img = svhn_img.to(config.device)
            labels = labels.to(config.device)

            if model.use_label_conditioning:
                output = model(mnist_img, svhn_img, labels)
            else:
                output = model(mnist_img, svhn_img)

            gamma = getattr(config, "gamma", 0.0)
            domain_loss_type = getattr(config, "domain_loss_type", "mmd")
            contrastive_weight = getattr(config, "contrastive_weight", 5.0)

            loss, recon_loss, kl_div, clf_loss, domain_loss, contrastive_loss = (
                vae_loss(
                    output["mnist_recon"],
                    output["svhn_recon"],
                    mnist_img,
                    svhn_img,
                    output["mu"],
                    output["logvar"],
                    beta=1.0,
                    free_bits=config.free_bits,
                    classifier_logits=output.get("classifier_logits"),
                    true_labels=labels,
                    z_mnist=output.get("z_mnist"),
                    z_svhn=output.get("z_svhn"),
                    z=output.get("z"),
                    gamma=gamma,
                    domain_loss_type=domain_loss_type,
                    svhn_recon_weight=config.svhn_recon_weight,
                    contrastive_weight=contrastive_weight,
                )
            )

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            total_domain_loss += (
                domain_loss if isinstance(domain_loss, float) else domain_loss.item()
            )
            total_contrastive_loss += (
                contrastive_loss
                if isinstance(contrastive_loss, float)
                else contrastive_loss.item()
            )

    avg_loss = total_loss / len(test_loader.dataset)
    avg_recon_loss = total_recon_loss / len(test_loader.dataset)
    avg_kl_div = total_kl_div / len(test_loader.dataset)
    avg_domain_loss = total_domain_loss / len(test_loader.dataset)
    avg_contrastive_loss = total_contrastive_loss / len(test_loader.dataset)

    return avg_loss, avg_recon_loss, avg_kl_div, avg_domain_loss, avg_contrastive_loss


def train_model(model, train_loader, test_loader, config):
    """
    Train the multi-view VAE model.

    Args:
        model: MultiViewVAE model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration object

    Returns:
        Trained model and training history
    """
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    history_csv_path = os.path.join(config.checkpoint_dir, "training_history.csv")

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_div": [],
        "train_domain_loss": [],
        "train_contrastive_loss": [],
        "test_loss": [],
        "test_recon_loss": [],
        "test_kl_div": [],
        "test_domain_loss": [],
        "test_contrastive_loss": [],
        "beta": [],
    }

    print(f"Training on device: {config.device}")
    print(f"Expert type: {config.expert_type}")
    print(f"Latent dim: {config.latent_dim}, Hidden dim: {config.hidden_dim}")
    print(
        f"Beta warmup: {config.beta_start} -> {config.beta_end} over {config.beta_warmup_epochs} epochs"
    )
    print(f"Free bits: {config.free_bits}")
    gamma = getattr(config, "gamma", 0.0)
    domain_loss_type = getattr(config, "domain_loss_type", "mmd")
    contrastive_weight = getattr(config, "contrastive_weight", 5.0)
    print(f"Domain loss: gamma={gamma}, type={domain_loss_type}")
    print(f"Contrastive loss weight: {contrastive_weight}")

    for epoch in range(1, config.num_epochs + 1):
        # train
        train_loss, train_recon, train_kl, train_domain, train_contrastive = (
            train_epoch(model, train_loader, optimizer, config, epoch)
        )

        # evaluate
        test_loss, test_recon, test_kl, test_domain, test_contrastive = evaluate(
            model, test_loader, config
        )

        scheduler.step(test_loss)

        if epoch <= config.beta_warmup_epochs:
            current_beta = config.beta_start + (config.beta_end - config.beta_start) * (
                epoch / config.beta_warmup_epochs
            )
        else:
            current_beta = config.beta_end

        history["train_loss"].append(train_loss)
        history["train_recon_loss"].append(train_recon)
        history["train_kl_div"].append(train_kl)
        history["train_domain_loss"].append(train_domain)
        history["train_contrastive_loss"].append(train_contrastive)
        history["test_loss"].append(test_loss)
        history["test_recon_loss"].append(test_recon)
        history["test_kl_div"].append(test_kl)
        history["test_domain_loss"].append(test_domain)
        history["test_contrastive_loss"].append(test_contrastive)
        history["beta"].append(current_beta)

        with open(history_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_recon_loss",
                    "train_kl_div",
                    "train_domain_loss",
                    "train_contrastive_loss",
                    "test_loss",
                    "test_recon_loss",
                    "test_kl_div",
                    "test_domain_loss",
                    "test_contrastive_loss",
                    "beta",
                ]
            )
            for i in range(len(history["train_loss"])):
                writer.writerow(
                    [
                        i + 1,
                        history["train_loss"][i],
                        history["train_recon_loss"][i],
                        history["train_kl_div"][i],
                        history["train_domain_loss"][i],
                        history["train_contrastive_loss"][i],
                        history["test_loss"][i],
                        history["test_recon_loss"][i],
                        history["test_kl_div"][i],
                        history["test_domain_loss"][i],
                        history["test_contrastive_loss"][i],
                        history["beta"][i],
                    ]
                )

        print(f"\nEpoch {epoch} (beta={current_beta:.4f}):")
        print(
            f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Domain: {train_domain:.4f}, Contr: {train_contrastive:.4f})"
        )
        print(
            f"  Test Loss:  {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f}, Domain: {test_domain:.4f}, Contr: {test_contrastive:.4f})"
        )
        print(f"  History exported to: {history_csv_path}")

        # save checkpoint
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"model_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"  Checkpoint saved: {checkpoint_path}")

    return model, history
