import os
import cv2
import time
import yaml
import random
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
from data_util import custom_to_pil, save_image
from vqgan_util import load_config, load_vqgan, vqgan_encoder
from dataset import TrainDataset
from model.restoration_network import RAHC
from model.discriminitor import NetD
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
from loss import L1Loss, BCELoss, SSIMLoss, CharbonnierLoss, EdgeLoss, PSNRLoss, ContrastLoss, CosLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# set random_seed
setup_seed(2023)


parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset', type=str, default='./datasets/HAC/train', help='dataset folder for train (change for different datasets)')
parser.add_argument('--val-dataset', type=str, default='./datasets/HAC/test', help='dataset forder for val')
parser.add_argument('--save-epoch-freq', type=int, default=5 , help='how often to save model')
parser.add_argument('--print-freq', type=int, default=20, help='how often to print training information')
parser.add_argument('--val', type=bool, default=False, help='val during training or not')
parser.add_argument('--val-freq', type=int, default=5, help='how often to val model')
parser.add_argument('--train-visual-freq', type=int, default=20, help='how often to visualize training process')
parser.add_argument('--val-visual-freq', type=int, default=30, help='how often to visualize validation process')
parser.add_argument('--resume', type=str, default=None, help='continue training from this checkpoint')
parser.add_argument('--start-epoch', type=int, default=1, help='start epoch')
parser.add_argument('--output-dir', type=str, default='./checkpoints', help='model saved folder')
parser.add_argument('--log-dir', type=str, default='./logs', help='save visual image')
parser.add_argument('--epochs', type=int, default=300, help='total number of epoch')
parser.add_argument('--image-size', type=int, default=256, help='image crop size')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='num of workers per GPU to use')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')


def build_model(opt):
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    config_vqgan = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
    vqgan, vqgan_dict = load_vqgan(config_vqgan, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True)
    vqgan = vqgan.cuda()

    discriminitor = NetD().cuda()

    model = RAHC().cuda()

    pretrained_dict = {k[9:]: v for k, v in vqgan_dict.items() if 'quantize' in k}
    model.mapping_network.quantize.load_state_dict(pretrained_dict, strict=False)

    for param in model.mapping_network.quantize.parameters():
        param.requires_grad = False
        param.retain_graph = True


    optimizer_rest = torch.optim.Adam(list(model.encoders.parameters())+
                                  list(model.decoders.parameters())+
                                  list(model.intro.parameters())+
                                  list(model.ending.parameters())+
                                  list(model.middle_pre.parameters())+
                                  list(model.middle_aft.parameters())+
                                  list(model.ups.parameters())+
                                  list(model.downs.parameters())+
                                  list(model.post_conv.parameters()),
                                lr=opt.lr,
                                 betas=(0.9, 0.999),eps=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_rest, opt.epochs,
                                                            eta_min=1e-6)

    optimizer_map = torch.optim.Adam(list(model.mapping_network.conv.parameters())+
                                     list(model.mapping_network.att.parameters()),
                                      lr=opt.lr,
                                      betas=(0.9, 0.999), eps=1e-8)

    optimizer_dis = torch.optim.Adam(discriminitor.parameters(),
                                     lr=opt.lr,
                                     betas=(0.9, 0.999), eps=1e-8)


    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    l1loss = L1Loss().cuda()
    bceloss = BCELoss().cuda()
    cosloss = CosLoss().cuda()
    loss = {'l1':l1loss, 'bce':bceloss, 'cos':cosloss}

    return model, vqgan, discriminitor, optimizer_rest, optimizer_map, optimizer_dis, loss, scheduler

def get_trainval_loader(opt):
    train_dataset = TrainDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers)

    return train_dataloader


def load_checkpoint(opt, model, optimizer):
    print(f"=> loading checkpoint '{opt.resume}'")

    checkpoint = torch.load(opt.resume, map_location='cpu')
    opt.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"=> loaded successfully '{opt.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(epoch, model, optimizer):
    print('==> Saving Epoch: {}'.format(epoch))
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(opt.output_dir, 'latest.pth'))



def train(epoch, model, vqgan, discriminitor, train_loader, optimizer_rest, optimizer_map, optimizer_dis, train_loss, opt):

    epoch_start_time = time.time()
    model.train()
    total_epoch_train_loss = []
    for batch_iter, data in enumerate(train_loader):
        image, target, label = data['in_img'].cuda(), data['gt_img'].cuda(), data['label'].cuda()

        output, f_rv = model(image)
# =====================discriminitor=============================
        optimizer_dis.zero_grad()

        dis_out_t = discriminitor(target)
        type = torch.zeros_like(label).cuda()
        dis_loss_t = train_loss['bce'](dis_out_t, type)
        dis_loss_t.backward()

        dis_out_d = discriminitor(output.detach())
        dis_loss_d = train_loss['bce'](dis_out_d, label)
        dis_loss_d.backward()

        optimizer_dis.step()
# =====================mapping network=============================
        vqgan_results = vqgan_encoder(target, vqgan).detach()
        optimizer_map.zero_grad()
        map_net_loss = train_loss['cos'](f_rv, vqgan_results)
        map_net_loss.backward()
        optimizer_map.step()

# =====================restoration network=============================
        optimizer_rest.zero_grad()
        dis_out_o = discriminitor(output)
        losses = train_loss['l1'](output, target) + 0.1 * train_loss['bce'](dis_out_o, type)
        losses.backward()
        optimizer_rest.step()


        total_epoch_train_loss.append(losses.cpu().data)

        if (batch_iter+1) % opt.print_freq == 0:
            print('Epoch: {}, Epoch_iter: {}, Loss: {}'.format(epoch, batch_iter+1, losses))
        if (batch_iter+1) % opt.train_visual_freq == 0:
            print('Saving training image epoch: {}'.format(epoch))
            save_image(epoch, 'train',[image, output, target] , opt)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.epochs, time.time() - epoch_start_time))

    return np.mean(total_epoch_train_loss) / opt.batch_size

def val(epoch, model, val_loader, val_loss, opt):
    model.eval()
    total_epoch_val_loss = []
    for i, data in enumerate(val_loader):
        image, target = data['in_img'].cuda(), data['gt_img'].cuda()

        output, _ = model(image)

        losses = val_loss['l1'](output, target)


        total_epoch_val_loss.append(losses.cpu().data)

        if (i+1) % opt.val_visual_freq == 0:
            print('Saving validation image epoch: {}'.format(epoch))
            save_image(epoch, 'val', [image, final, target], opt)
    return np.mean(total_epoch_val_loss) / opt.batch_size

def main(opt):

    model, vqgan, discriminitor, optimizer_rest, optimizer_map, optimizer_dis, loss, scheduler = build_model(opt)
    scheduler.step()
    if opt.val:
        train_loader, val_loader = get_trainval_loader(opt)
    else:
        train_loader = get_trainval_loader(opt)

    total_train_loss = []
    total_val_loss = []

    if opt.resume:
        assert os.path.isfile(opt.resume)
        load_checkpoint(opt, model, optimizer_rest)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        epoch_train_loss = train(epoch, model, vqgan, discriminitor, train_loader, optimizer_rest, optimizer_map, optimizer_dis, loss, opt)
        total_train_loss.append(epoch_train_loss)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % epoch)
            save_checkpoint(epoch, model, optimizer_rest)

        if opt.val and epoch % opt.val_freq == 0:
            epoch_val_loss = val(epoch, model, val_loader, loss, opt)
            print('Epoch: {}, Validation_Loss: {}'.format(epoch, epoch_val_loss))
            total_val_loss.append(epoch_val_loss)
        scheduler.step()

    plt.subplot(211)
    plt.plot(total_train_loss)
    plt.subplot(212)
    plt.plot(total_val_loss)
    plt.savefig(opt.log_dir + '/loss.png')

if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)

