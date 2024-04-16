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

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR10 dataset
dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 100)

        self.model = nn.Sequential(
            nn.Linear(100 + 100, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 4*4*1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4*4*1024),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((label_input, noise), dim=1)
        img = self.model(gen_input)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 64*64)  

        self.model = nn.Sequential(
            nn.Linear(3*64*64 + 64*64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
        )
        
        self.validity_layer = nn.Linear(128, 1)
        self.aux_label_layer = nn.Linear(128, 10)

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_emb(labels).view(img.size(0), -1)
        d_in = torch.cat((img_flat, label_input), dim=1)
        features = self.model(d_in)
        validity = self.validity_layer(features)
        label = self.aux_label_layer(features)
        return validity, label


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

criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=5e-5, betas=(0.5, 0.999))

inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()
inception_model.eval().to(device)

def inception_preprocess():
    return transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_inception_features(images, model):
    images = inception_preprocess()(images).to('cuda')
    with torch.no_grad():
        features = model(images).detach().cpu()
    return features.numpy()

def calculate_fid_score(features_real, features_fake):
    mu1, sigma1 = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
    mu2, sigma2 = features_fake.mean(axis=0), np.cov(features_fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

num_epochs = 50
fixed_noise = torch.randn(64, 100, 1, 1, device=device)
g_losses = []
d_losses = []
fid_scores = []
iters = 0
adversarial_loss = torch.nn.BCEWithLogitsLoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_idx, (imgs,labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        real_imgs = imgs.to(device)
        labels = labels.to(device)

        z = torch.randn(batch_size, 100, device=device)
        gen_labels = torch.randint(0, 10, (batch_size,), dtype=torch.long, device=device)

        generated_imgs = generator(z, gen_labels)

        optimizerD.zero_grad()

        real_validity, real_aux = discriminator(real_imgs, labels)
        d_real_loss = (adversarial_loss(real_validity, valid) + auxiliary_loss(real_aux, labels)) / 2

        fake_validity, fake_aux = discriminator(generated_imgs.detach(), gen_labels)
        d_fake_loss = (adversarial_loss(fake_validity, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizerD.step()

        optimizerG.zero_grad()

        validity, aux = discriminator(generated_imgs, gen_labels)
        g_loss = (adversarial_loss(validity, valid) + auxiliary_loss(aux, gen_labels)) / 2
        g_loss.backward()
        optimizerG.step()

        if batch_idx % 100 == 0:
            print(f'[{epoch+1}/{num_epochs}][{batch_idx}/{len(dataloader)}] Loss_D: {g_loss:.4f} Loss_G: {d_loss:.4f}')

        iters += 1
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    fid_features_real = get_inception_features(real_imgs, inception_model)
    fid_features_fake = get_inception_features(generated_imgs, inception_model)
    fid_score = calculate_fid_score(fid_features_real, fid_features_fake)
    fid_scores.append(fid_score)
    print(f'[{epoch+1}/{50}] FID: {fid_score:.4f}')

    save_image(generated_imgs, './figure/acgan/fake/epoch_%03d.png' % (epoch), nrow=8, normalize=True)
    save_image(real_imgs, './figure/acgan/real/epoch_%03d.png' % (epoch), nrow=8, normalize=True)


plt.figure(figsize=(10,5))
plt.title("acgan Loss During Training: Generator and Discriminator")
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(np.arange(0,50), fid_scores, label = 'acgan')
plt.title('Fretchet Distance for 50 epochs')
plt.legend()
plt.show()

def save_list_to_txt(list_data, file_path):
    with open(file_path, 'w') as file:
        for item in list_data:
            file.write(f"{item}\n")

g_losses_path = './result/acgan_g_losses.txt' 
d_losses_path = './result/acgan_d_losses.txt'
fid_scores_path = './result/acgan_fid_scores.txt'
save_list_to_txt(g_losses, g_losses_path)
save_list_to_txt(d_losses, d_losses_path)
save_list_to_txt(fid_scores, fid_scores_path)