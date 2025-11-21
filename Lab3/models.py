import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization for FiLM-style (Feature-wise Linear Modulation) conditioning.
    Applies label-dependent affine transformations to feature maps.
    """

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.gamma_embed = nn.Linear(num_classes, num_features)
        self.beta_embed = nn.Linear(num_classes, num_features)

        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.gamma_embed.bias)
        nn.init.zeros_(self.beta_embed.weight)
        nn.init.zeros_(self.beta_embed.bias)

    def forward(self, x, labels_onehot):
        """
        Args:
            x: Feature maps [B, C, H, W]
            labels_onehot: One-hot encoded labels [B, num_classes]
        """
        out = self.bn(x)

        # label-dependent affine parameters
        gamma = self.gamma_embed(labels_onehot).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(labels_onehot).view(-1, self.num_features, 1, 1)

        # FiLM transformation: gamma * x + beta
        return gamma * out + beta


class Encoder(nn.Module):
    def __init__(
        self, input_channels, input_size, hidden_dim, latent_dim, num_classes=0
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            # input: [B, input_channels, 32, 32]
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # output: [B, 32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # output: [B, 64, 8, 8]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # output: [B, 128, 4, 4]
        )

        conv_output_size = input_size // 8
        self.conv_output_dim = 128 * conv_output_size * conv_output_size

        fc_input_dim = (
            self.conv_output_dim + num_classes
            if num_classes > 0
            else self.conv_output_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, labels=None):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        if self.num_classes > 0 and labels is not None:
            batch_size = x.size(0)
            labels = labels.to(x.device)
            label_onehot = torch.zeros(batch_size, self.num_classes, device=x.device)
            label_onehot.scatter_(1, labels.unsqueeze(1), 1)
            x = torch.cat([x, label_onehot], dim=1)

        x = self.fc(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class MNISTEncoder(Encoder):
    def __init__(self, hidden_dim, latent_dim, num_classes=0):
        super().__init__(
            input_channels=1,
            input_size=32,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
        )


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels, num_classes=0):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.output_size = 32

        self.deconv_input_size = 4
        self.deconv_input_dim = 128 * self.deconv_input_size * self.deconv_input_size

        fc_input_dim = latent_dim + num_classes if num_classes > 0 else latent_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.deconv_input_dim),
            nn.ReLU(),
        )

        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.cbn1 = ConditionalBatchNorm2d(128, num_classes)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.cbn2 = ConditionalBatchNorm2d(128, num_classes)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.cbn3 = ConditionalBatchNorm2d(64, num_classes)
        self.relu3 = nn.ReLU()

        self.refine = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.cbn4 = ConditionalBatchNorm2d(32, num_classes)
        self.relu4 = nn.ReLU()

        self.final = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def forward(self, z, labels=None):
        if self.num_classes > 0 and labels is not None:
            batch_size = z.size(0)
            labels = labels.to(z.device)
            label_onehot = torch.zeros(batch_size, self.num_classes, device=z.device)
            label_onehot.scatter_(1, labels.unsqueeze(1), 1)
            z = torch.cat([z, label_onehot], dim=1)

        x = self.fc(z)
        x = x.view(x.size(0), 128, self.deconv_input_size, self.deconv_input_size)

        if self.num_classes > 0 and labels is not None:
            x = self.deconv1(x)
            x = self.cbn1(x, label_onehot)
            x = self.relu1(x)

            x = self.deconv2(x)
            x = self.cbn2(x, label_onehot)
            x = self.relu2(x)

            x = self.deconv3(x)
            x = self.cbn3(x, label_onehot)
            x = self.relu3(x)

            x = self.refine(x)
            x = self.cbn4(x, label_onehot)
            x = self.relu4(x)

            x = self.final(x)
            x = self.activation(x)
        else:
            x = self.deconv_layers(x)

        return x


class MNISTDecoder(Decoder):
    def __init__(self, latent_dim, hidden_dim, num_classes=0):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_channels=1,
            num_classes=num_classes,
        )


class SVHNEncoder(Encoder):
    def __init__(self, hidden_dim, latent_dim, num_classes=0):
        super().__init__(
            input_channels=3,
            input_size=32,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
        )


class SVHNDecoder(Decoder):
    def __init__(self, latent_dim, hidden_dim, num_classes=0):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_channels=3,
            num_classes=num_classes,
        )
