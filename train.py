import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator
from loss import calcul_gradient_penalty
from utils import make_exp_dirs, read_img, get_device, create_grid, set_require_grads

EXP_NAME = 'poc'
IMG_PATH = 'inputs/small_balloon.png'

Z_DIM = 128
SCALE = 10
MAPPING_SIZE = 256

LR = 1e-4
BATCH_SIZE = 16
MAX_ITERS = 1000000

N_CRITIC = 5
GEN_ITER = 1
GP_LAMBDA = 0.1
RECON_LAMBDA = 10.


if __name__ == '__main__':
    writer = make_exp_dirs(EXP_NAME, log=True)

    img = read_img(IMG_PATH)
    h, w, _ = img.shape

    device = get_device()

    B_gauss = torch.randn((MAPPING_SIZE, 2)).to(device) * SCALE
    torch.save(B_gauss, f'exps/{EXP_NAME}/ckpt/B.pth')

    grid = create_grid(h, w, device)
    mapped_input = torch.sin((2. * np.pi * grid) @ B_gauss.t())
    mapped_input = mapped_input.repeat(BATCH_SIZE, 1, 1, 1).view(BATCH_SIZE, h*w, MAPPING_SIZE)

    recon_z = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
    reals = torch.FloatTensor(img).permute(2, 0, 1).repeat(BATCH_SIZE, 1, 1, 1).to(device)


    generator = Generator(input_dim=MAPPING_SIZE, z_dim=Z_DIM, hidden_dim=256, output_dim=3).to(device)
    discriminator = Discriminator(in_channels=3, max_features=32, min_features=32, num_blocks=5).to(device)
    g_optim = torch.optim.Adam(generator.parameters(), lr=LR)
    d_optim = torch.optim.Adam(generator.parameters(), lr=LR)
    g_scheduler = StepLR(g_optim, step_size=2000, gamma=0.1)
    d_scheduler = StepLR(d_optim, step_size=2000, gamma=0.1)

    set_require_grads(generator, True)
    recon_criterion = torch.nn.MSELoss()
    for iter in range(MAX_ITERS):

        # Train Discriminator
        for i in range(N_CRITIC):
            set_require_grads(discriminator, True)

            z = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
            generated = generator(grid, z).view(BATCH_SIZE, h, w, 3).permute(0, 3, 1, 2)

            d_optim.zero_grad()
            d_generated = discriminator(generated)
            d_real = discriminator(reals)

            loss_r = -d_real.mean()
            loss_f = d_generated.mean()
            gradient_penalty = calcul_gradient_penalty(discriminator, reals, generated, device) * GP_LAMBDA
            d_loss = loss_r + loss_f + gradient_penalty

            d_loss.backward()
            d_optim.step()

            set_require_grads(discriminator, False)

        critic = - loss_r - loss_f
        writer.add_scalar("d/total", d_loss.item(), iter)
        writer.add_scalar("d/critic - max", critic.item(), iter)
        writer.add_scalar("d/gp", gradient_penalty.item(), iter)

        for i in range(GEN_ITER):
            g_optim.zero_grad()

            recon = generator(mapped_input, recon_z).view(BATCH_SIZE, h, w, 3).permute(0, 3, 1, 2)
            loss_recon = recon_criterion(recon, reals) * RECON_LAMBDA

            adv_z = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
            generated = generator(grid, adv_z).view(BATCH_SIZE, h, w, 3).permute(0, 3, 1, 2)
            d_generated = discriminator(generated)
            loss_adv = -d_generated.mean()

            g_loss = loss_adv + loss_recon
            g_loss.backward()
            g_optim.step()

        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic - min", loss_adv.item(), iter)
        writer.add_scalar("g/recon", loss_recon.item(), iter)

        writer.flush()

        g_scheduler.step(epoch=iter)
        d_scheduler.step(epoch=iter)

        if iter % 50 == 0:
            recon_idx = np.random.randint(high=BATCH_SIZE)
            save_image(recon[recon_idx], f'exps/{EXP_NAME}/img/{iter}_recon.jpg')
            save_image(generated[0], f'exps/{EXP_NAME}/img/{iter}_adv.jpg')

        if iter % 1000 == 0:
            torch.save(generator.state_dict(), f'exps/{EXP_NAME}/ckpt/G.pth')
            torch.save(discriminator.state_dict(), f'exps/{EXP_NAME}/ckpt/D.pth')