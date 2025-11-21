import torch
import torch.nn as nn
from models import MNISTEncoder, MNISTDecoder, SVHNEncoder, SVHNDecoder


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def product_of_experts(mu_list, logvar_list, eps=1e-8):
    """
    Returns:
        mu_poe: Combined mean
        logvar_poe: Combined log variance
    """
    # log variance to precision (inverse variance)
    var_list = [torch.exp(logvar) + eps for logvar in logvar_list]
    precision_list = [1.0 / var for var in var_list]

    # PoE formula
    precision_poe = torch.sum(torch.stack(precision_list), dim=0)
    var_poe = 1.0 / precision_poe

    mu_poe = var_poe * torch.sum(
        torch.stack([mu * precision for mu, precision in zip(mu_list, precision_list)]),
        dim=0,
    )

    logvar_poe = torch.log(var_poe)

    return mu_poe, logvar_poe


def mixture_of_experts(mu_list, logvar_list):
    """
    Returns:
        mu_moe: Combined mean
        logvar_moe: Combined log variance
    """
    n_experts = len(mu_list)
    weights = 1.0 / n_experts

    # simple average for MoE with uniform weights
    mu_moe = weights * torch.sum(torch.stack(mu_list), dim=0)
    logvar_moe = weights * torch.sum(torch.stack(logvar_list), dim=0)

    return mu_moe, logvar_moe


class MultiViewVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.expert_type = config.expert_type
        self.use_label_conditioning = getattr(config, "use_label_conditioning", False)
        self.num_classes = (
            getattr(config, "num_classes", 10) if self.use_label_conditioning else 0
        )

        # encoders
        self.mnist_encoder = MNISTEncoder(
            self.hidden_dim, self.latent_dim, self.num_classes
        )
        self.svhn_encoder = SVHNEncoder(
            self.hidden_dim, self.latent_dim, self.num_classes
        )

        # decoders
        self.mnist_decoder = MNISTDecoder(
            self.latent_dim, self.hidden_dim, self.num_classes
        )
        self.svhn_decoder = SVHNDecoder(
            self.latent_dim, self.hidden_dim, self.num_classes
        )

        # classifier for label informative latent codes
        if self.use_label_conditioning:
            self.classifier = nn.Linear(self.latent_dim, self.num_classes)

    def encode_mnist(self, x, labels=None):
        return self.mnist_encoder(x, labels)

    def encode_svhn(self, x, labels=None):
        return self.svhn_encoder(x, labels)

    def decode_mnist(self, z, labels=None):
        return self.mnist_decoder(z, labels)

    def decode_svhn(self, z, labels=None):
        return self.svhn_decoder(z, labels)

    def combine_experts(self, mu_list, logvar_list):
        if self.expert_type == "poe":
            return product_of_experts(mu_list, logvar_list)
        elif self.expert_type == "moe":
            return mixture_of_experts(mu_list, logvar_list)
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")

    def forward(self, mnist_img=None, svhn_img=None, labels=None):
        """
        Returns:
            Dictionary containing reconstructions and latent parameters
        """
        mu_list = []
        logvar_list = []
        z_mnist = None
        z_svhn = None

        if mnist_img is not None:
            mu_mnist, logvar_mnist = self.encode_mnist(mnist_img, labels)
            mu_list.append(mu_mnist)
            logvar_list.append(logvar_mnist)
            z_mnist = reparameterize(mu_mnist, logvar_mnist)

        if svhn_img is not None:
            mu_svhn, logvar_svhn = self.encode_svhn(svhn_img, labels)
            mu_list.append(mu_svhn)
            logvar_list.append(logvar_svhn)
            z_svhn = reparameterize(mu_svhn, logvar_svhn)

        if len(mu_list) > 1:
            mu, logvar = self.combine_experts(mu_list, logvar_list)
        elif len(mu_list) == 1:
            mu, logvar = mu_list[0], logvar_list[0]
        else:
            raise ValueError("At least one modality must be provided")

        z = reparameterize(mu, logvar)

        mnist_recon = self.decode_mnist(z, labels)
        svhn_recon = self.decode_svhn(z, labels)

        result = {
            "mnist_recon": mnist_recon,
            "svhn_recon": svhn_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_mnist": z_mnist,
            "z_svhn": z_svhn,
        }

        if self.use_label_conditioning:
            result["classifier_logits"] = self.classifier(z)

        return result

    def generate(self, num_samples, device, labels=None):
        """
        Returns:
            Dictionary containing generated MNIST and SVHN images
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)

        if self.use_label_conditioning and labels is None:
            labels = torch.zeros(num_samples, dtype=torch.long, device=device)

        mnist_gen = self.decode_mnist(z, labels)
        svhn_gen = self.decode_svhn(z, labels)

        return {"mnist_gen": mnist_gen, "svhn_gen": svhn_gen, "z": z}

    def cross_generate(self, source_img, source_type="mnist", labels=None):
        """
        Returns:
            Dictionary containing cross-generated images
        """
        if self.use_label_conditioning and labels is None:
            batch_size = source_img.size(0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=source_img.device)

        if source_type == "mnist":
            mu, logvar = self.encode_mnist(source_img, labels)
            z = mu  # deterministic inference -> more consistent cross-generation
            target_recon = self.decode_svhn(z, labels)
            return {"mnist_to_svhn": target_recon, "z": z, "mu": mu, "logvar": logvar}
        elif source_type == "svhn":
            mu, logvar = self.encode_svhn(source_img, labels)
            z = mu  # deterministic inference -> more consistent cross-generation
            target_recon = self.decode_mnist(z, labels)
            return {"svhn_to_mnist": target_recon, "z": z, "mu": mu, "logvar": logvar}
        else:
            raise ValueError(f"Unknown source type: {source_type}")
