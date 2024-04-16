import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as utils
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR10 dataset
dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameters
latent_dim = 100
batch_size = 64
channels = 3
lr = 2e-4
n_discriminator = 5
clip_value = 0.01  # clip parameter for WGAN
epochs = 50
img_size = 64

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

# Critic (Discriminator) Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.model(input).view(-1, 1)

# Initialize generator and critic
generator = Generator().to(device)
discriminator = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers
criterion = nn.BCELoss()
optimizerG = optim.RMSprop(generator.parameters(), lr=lr)
optimizerD = optim.RMSprop(discriminator.parameters(), lr=lr)

generator.train()
discriminator.train()

# Load the pre-trained Inception v3 model
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  
inception_model = inception_model.to('cuda')
inception_model.eval()

# Function for preprocessing images for the Inception v3 model
def inception_preprocess():
    return transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Function to calculate features using Inception model
def get_inception_features(images, model):
    images = inception_preprocess()(images).to('cuda')
    with torch.no_grad():
        features = model(images).detach().cpu()
    return features.numpy()

# Function to calculate the FID score
def calculate_fid_score(features_real, features_fake):
    mu1, sigma1 = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
    mu2, sigma2 = features_fake.mean(axis=0), np.cov(features_fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check for any numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

epochs = 50
iters = 0
fixed_noise = torch.randn(32, 100, 1, 1).to(device)
img_list = []
g_losses = []
d_losses = []
fid_scores = []

for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train Critic with real images
        discriminator.zero_grad()
        real_images = real_images.to(device)
        real_loss = discriminator(real_images).mean()
        real_loss.backward(torch.tensor(-1.0))  # mone is torch.tensor(-1.0), which is used to flip the sign

        # Train Critic with fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise).detach()  # detach to avoid training the generator on these labels
        fake_loss = discriminator(fake_images).mean()
        fake_loss.backward(torch.tensor(1.0))  # one is torch.tensor(1.0)

        discriminator_loss = fake_loss - real_loss
        optimizerD.step()

        # Clip weights of critic to satisfy Lipschitz constraint
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Update Generator
        if i % n_discriminator == 0:
            generator.zero_grad()
            gen_fake = generator(noise)
            gen_loss = -discriminator(gen_fake).mean()
            gen_loss.backward()
            optimizerG.step()

            g_losses.append(gen_loss.item())
            d_losses.append(discriminator_loss.item())

            print(f'[{epoch+1}/{epochs}][{i}/{len(dataloader)}] discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}')

        if i % 100 == 0:
            # Save images to check generator's performance
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    # Optionally calculate FID score here, similar to previous setup
    generator.eval()
    fid_features_real = get_inception_features(real_images, inception_model)
    fid_features_fake = get_inception_features(fake_images, inception_model)
    fid_score = calculate_fid_score(fid_features_real, fid_features_fake)
    fid_scores.append(fid_score)
    
    print(f'[{epoch+1}/{epochs}] FID: {fid_score:.4f}')

    # Save some generated images during training
    with torch.no_grad():
        fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
        fake_images = generator(fixed_noise).detach().cpu()
    save_image(fake_images, f'./figure/wgan/fake/epoch_{epoch:03d}.png', nrow=8, normalize=True)
    save_image(real_images, f'./figure/wgan/real/epoch_{epoch:03d}.png', nrow=8, normalize=True)


plt.figure(figsize=(10,5))
plt.title("wgan Loss During Training: Generator and Discriminator")
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(np.arange(0,50), fid_scores, label = 'wgan')
plt.title('Fretchet Distance for 50 epochs')
plt.legend()
plt.show()

def save_list_to_txt(list_data, file_path):
    with open(file_path, 'w') as file:
        for item in list_data:
            file.write(f"{item}\n")

g_losses_path = './result/wgan_g_losses.txt' 
d_losses_path = './result/wgan_d_losses.txt'
fid_scores_path = './result/wgan_fid_scores.txt'
save_list_to_txt(g_losses, g_losses_path)
save_list_to_txt(d_losses, d_losses_path)
save_list_to_txt(fid_scores, fid_scores_path)