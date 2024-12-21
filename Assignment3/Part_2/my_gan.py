import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    loss_function = torch.nn.BCELoss()
    for epoch in range(config.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # imgs.cuda()
            real_img = imgs.view(imgs.size(0), -1).to(device)
            real_img_label = torch.ones((imgs.size(0), 1), device=device, dtype=torch.float)
            fake_img_label = torch.zeros((imgs.size(0), 1), device=device, dtype=torch.float)

            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(imgs.size(0), config.latent_dim, device=device)
            gen_img = generator(noise)
            g_loss = loss_function(discriminator(gen_img), real_img_label)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = loss_function(discriminator(real_img), real_img_label)
            fake_loss = loss_function(discriminator(gen_img.detach()), fake_img_label)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        # Save Images
        # batches_done = epoch * len(dataloader) + i
        # if batches_done % config.save_interval == 0:
        if epoch % config.save_interval == 0 or epoch == config.n_epochs - 1:
            # You can use the function save_image(Tensor (shape Bx1x28x28),
            # filename, number of rows, normalize) to save the generated
            # images, e.g.:
            # save_image(gen_imgs[:25],
            #            'images/{}.png'.format(batches_done),
            #            nrow=5, normalize=True, value_range=(-1,1))
            noise = torch.randn(25, config.latent_dim, device=device)
            gen_img = generator(noise)
            save_image(
                gen_img.view(25, 1, 28, 28),
                f"images/{epoch}.png",
                nrow=5, normalize=True, value_range=(-1, 1))


def main(config):
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5),
                                                (0.5))])),
        batch_size=config.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(config.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, config)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=100,  # changed here
                        help='save every SAVE_INTERVAL iterations')
    config = parser.parse_args()

    main(config)

