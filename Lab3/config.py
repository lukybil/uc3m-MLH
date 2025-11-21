import torch


class Config:
    # device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data configuration
    data_dir = "./data"
    batch_size = 128
    num_workers = 0  # 0 for windows because of dataloader issues

    # model configuration
    latent_dim = 192
    hidden_dim = 800
    use_label_conditioning = True
    num_classes = 10

    # mnist configuration
    mnist_channels = 1
    mnist_size = 28

    # svhn configuration
    svhn_channels = 3
    svhn_size = 32

    # training configuration
    num_epochs = 50
    learning_rate = 1e-3

    # beta scheduling for kl divergence
    beta_start = 0.0
    beta_end = 1.0
    beta_warmup_epochs = 30

    gamma = 5.0  # weight for domain loss

    free_bits = 0.2  # minimum kl divergence per latent dimension

    contrastive_weight = (
        5.0  # weight for supervised contrastive loss (improves clustering)
    )

    svhn_recon_weight = (
        1.5  # multiplier for svhn reconstruction loss (to balance rgb vs grayscale)
    )

    expert_type = "poe"  # 'poe' or 'moe'

    # visualization configuration
    num_samples = 10
    tsne_perplexity = 30

    # checkpoint configuration
    checkpoint_dir = "./checkpoints"
    save_interval = 10

    # results configuration
    results_dir = "./results"
