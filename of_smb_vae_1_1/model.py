import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# vae 1
class Encoder(nn.Module):
    def __init__(self, num_features=7):
        super(Encoder, self).__init__()
        # Encoder
        self.cnv1 = nn.Conv2d(num_features, 8, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.cnv2 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)


    def forward(self, x):
        x = F.leaky_relu(self.batch_norm1(self.cnv1(x)))
        x = F.leaky_relu(self.batch_norm2(self.cnv2(x)))
        x = x.reshape(-1, 64)
        return self.fc1(x), self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, num_features=7):
        super(Decoder, self).__init__()
        # Decoder
        self.fc1 = nn.Linear(64,64)
        self.cnv1 = nn.ConvTranspose2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.cnv2 = nn.ConvTranspose2d(8, num_features, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 4, 4, 4)
        x = F.relu(self.batch_norm1(self.cnv1(x)))
        x = self.cnv2(x)
        x = F.softmax(x,dim=1)
        return x
        
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        y = self.reparameterize(mu, logvar)
        z = self.decoder(y)
        return z, mu, logvar