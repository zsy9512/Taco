# 与 Encodec_24k_240d main3_ddp.py 相比只有鉴别器不同
import argparse
import itertools
import os
import sys
import time
import torch
from src.modules.net import TouchStream
from tqdm import tqdm
from src.utils.dataset import PWQDataset_SMR , PWQDataset
from src.utils.loss import loss,criterion,Wfft_loss,fft_loss
from src.utils.utils import Logger
from src.utils.utils import seed_everything
from tqdm import tqdm
#from vector_quantize_pytorch import ResidualVQ,VectorQuantize,GroupedResidualVQ
from src.quantization import ResidualVectorQuantizer,GroupVectorQuantizer
"""
    python /home/zsy/workplace/taco/train_vq.py  \
    --BATCH_SIZE 32 \
    --N_EPOCHS 500 \
    --seed 123 \
    --save_dir /home/zsy/workplace/taco/output/logs  \
    --PATH  /home/zsy/workplace/taco/output/checkpoints \
    --train_data_path /home/zsy/workplace/taco/dataset/pwq_or/train \
    --valid_data_path /home/zsy/workplace/taco/dataset/pwq_or/val \
    --log \
    --tensorboard \
    --group_num 2 \
    --bins 256 \
    --nq 1
"""
def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='GPU id to use. If given, only the specific gpu will be'
        ' used, and ddp will be disabled')
    parser.add_argument(
        '--device_rank',
        default=0,
        type=int,
        help='node rank for distributed training')
    # args for random
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='seed for initializing training. ')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='use tensorboard for logging')
    parser.add_argument(
        '--log',
        action='store_true',
        help='logging files')
    # args for training
    parser.add_argument(
        '--LAMBDA_REC',
        type=float,
        default=1.0,
        help='hyper-parameter for rec loss')
    parser.add_argument(
        '--LAMBDA_COM',
        type=float,
        default=100,
        help='hyper-parameter for commit loss')
    parser.add_argument(
        '--LAMBDA_SPEC',
        type=float,
        default=0.00,
        help='hyper-parameter for spec loss')
    parser.add_argument(
        '--N_EPOCHS', type=int, default=100, help='Total training epoch')
    parser.add_argument(
        '--st_epoch', type=int, default=0, help='start training epoch')
    parser.add_argument(
        '--global_step', type=int, default=0, help='record the global step')                   
    parser.add_argument('--BATCH_SIZE', type=int, default=2, help='batch size')
    parser.add_argument(
        '--PATH',
        type=str,
        default='model_path/',
        help='The path to save the model')
    parser.add_argument('--sr', type=int, default=2800, help='sample rate')
    parser.add_argument(
        '--print_freq', type=int, default=501, help='the print number')
    parser.add_argument(
        '--save_dir', type=str, default='log', help='log save path')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='path_to_wavs',
        help='training data')
    parser.add_argument(
        '--valid_data_path',
        type=str,
        default='path_to_val_wavs',
        help='training data')
    parser.add_argument(
        '--resume', action='store_true', help='whether re-train model')
    parser.add_argument(
        '--resume_path', type=str, default='path_to_resume', help='resume_path')
    parser.add_argument(
        '--bins', type=int, default=256, help='codebook size')
    parser.add_argument(
        '--nq',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--group_num',
        type=int,
        default=2,
    )
    args = parser.parse_args()
    # train_set=[{1, 64}, {2, 64}, {1, 128},{2, 128},{3, 128},
    #            {1, 256},{2, 256},{3, 256},{4, 256},{5, 256},
    #            {6, 256}, {8, 256},{10, 256},{16, 256},{1, 512},
    #            {2, 512},{3, 512},{14, 512},{1, 1024},{16,1024}]
    train_set=[{1, 256},{2, 256},{4, 256}]
    check={args.nq,args.bins}
    if check not in train_set:
        print("nq_"+str(int(args.nq))+"_"+str(int(args.bins))+" model is skip")
        sys.exit()
    else:
        print("nq_"+str(int(args.nq))+"_"+str(int(args.bins))+" model is train")
    #time_str = time.strftime('%Y-%m-%d-%H-%M')+"_nq_"+str(int(args.nq))+"_spec_"+str(int(args.LAMBDA_SPEC))
    time_str ="nq_"+str(int(args.nq))+"_"+str(int(args.bins))
    if args.resume:
        args.PATH = args.resume_path  # direcly use the old model path
    else:
        args.PATH = os.path.join(args.PATH, time_str)
    args.save_dir = os.path.join(args.save_dir, time_str)
    if args.log:
        os.makedirs(args.PATH, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
    return args


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()
def get_input_split(x):
    x = x.to(memory_format=torch.contiguous_format).float()
    x=torch.cat([x,torch.zeros(x.shape[0],x.shape[1],512*6-x.shape[2]).to("cuda")],dim=2)
    y=torch.split(x,512,dim=2)
    return x,y


def main():
    import numpy as np
    import math
    args = get_args()
    logger = Logger(args)
    bins=args.bins
    n_q=args.nq
    n_group=args.group_num
    if args.group_num>1 and n_q>1:#使用分组量化
        quantizer =  GroupVectorQuantizer(dimension=128, codebook_dim=128,n_q=n_q, bins=bins ,n_codes_group=n_group)
    else:
        quantizer = ResidualVectorQuantizer(dimension=128, codebook_dim=128,n_q=n_q, bins=bins)
    model= TouchStream(input_dim=128, quantizer=quantizer)
    getModelSize(model)
    args.device = torch.device('cuda', args.device_rank)
    model.to(args.device)
    train_dataset = PWQDataset_SMR(file_dir=args.train_data_path,max_len=512)
    valid_dataset = PWQDataset_SMR(file_dir=args.valid_data_path,max_len=512)
    
    args.sr = train_dataset.sr
    train_sampler = None
    valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=16,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=16,
        sampler=valid_sampler)
    optimizer_g = torch.optim.AdamW(model.parameters(),lr=0.001 , betas=(0.5, 0.9))
    lr_scheduler_g=torch.optim.lr_scheduler.StepLR(optimizer_g,step_size=1,gamma=0.99)
    optimizer_q = torch.optim.AdamW([{'params':model.quantizer.parameters(),'lr':0.00025},
                                    {'params':model.encoder.blocks.parameters(),'lr':0.0005},
                                     {'params':model.decoder.blocks.parameters(),'lr':0.0005},], betas=(0.5, 0.9))
    #optimizer_q=torch.optim.AdamW(model.parameters(),lr=0.0001 , betas=(0.5, 0.9))
    lr_scheduler_q =torch.optim.lr_scheduler.StepLR(optimizer_q,step_size=1,gamma=0.99)
    #lr_scheduler_q=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_q, mode='max', factor=0.8, patience=6, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    if args.resume:
        latest_info = torch.load(args.resume_path + '/latest.pth')
        model.load_state_dict(latest_info['model'],strict=False)
        optimizer_g.load_state_dict(latest_info['optimizer_g'])
        lr_scheduler_g.load_state_dict(latest_info['lr_scheduler_g'])

    train(args, model,  train_loader, valid_loader,
          optimizer_g,lr_scheduler_g,optimizer_q ,lr_scheduler_q, logger)
def train(args, model,  train_loader, valid_loader,
          optimizer_g, lr_scheduler_g,optimizer_q ,lr_scheduler_q, logger):
    #import auraloss
    best_val_loss = 0
    best_val_epoch = -1
    global_step = 0
    #stft = auraloss.freq.STFTLoss(fft_size=512,hop_size=32,win_length=512,reduction='none')
    count=0
    for epoch in range(args.st_epoch, args.N_EPOCHS + 1):
        model.train()
        train_rec_loss = 0.0
        train_loss_g = 0.0
        train_vq_loss = 0.0
        train_spec_loss = 0.0
        train_psnr=0.0
        k_iter = 0
        is_quan_on_val = 0.0 if epoch < 0 else 1.0
        for x,smr,maskenerge in tqdm(train_loader):
            x = x.to(args.device)
            smr=smr.to(args.device)
            maskenerge=maskenerge.to(args.device)
            k_iter += 1
            global_step += 1  # record the global step
            x_wav = get_input(x)
            G_x,vq_loss= model(x_wav,is_quan_on_val)
            rec_loss,psnr= loss(
                        x_wav,
                        G_x,
                        args=args)
            spec_loss=Wfft_loss(x,G_x,smr)
            vq_loss=vq_loss
            train_vq_loss += vq_loss.item()
            train_spec_loss += spec_loss.item()
            if is_quan_on_val:
                total_loss_g=args.LAMBDA_REC*rec_loss+args.LAMBDA_COM*vq_loss+args.LAMBDA_SPEC*spec_loss
            else:
                total_loss_g=rec_loss
            train_loss_g += total_loss_g.item()
            train_rec_loss += rec_loss.item()
            train_psnr +=psnr
            optimizer_q.zero_grad()
            optimizer_g.zero_grad()
            total_loss_g.backward()
            if is_quan_on_val:
                optimizer_q.step()
            else:
                optimizer_g.step()
            message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.6f},rec_loss:{:.6f},vq_loss:{:.5f},spec_loss:{:.5f}>'.format(
                epoch, k_iter,
                rec_loss.item(),
                psnr.item(),
                vq_loss.item(),
                spec_loss.item())
            if k_iter % args.print_freq == 0:
                logger.log_info(message)
        message = '<epoch:{:d}, <total_loss_g_train:{:.6f}, train_psnr:{:.6f}, vq_loss_train:{:.5f}, spec_loss_train:{:.5f}>'.format(
            epoch, train_loss_g / len(train_loader),
            train_psnr /len(train_loader), 
            train_vq_loss / len(train_loader),
            train_spec_loss / len(train_loader))
        logger.add_scalar(tag='train/spec_loss', scalar_value=(train_spec_loss /len(train_loader)), global_step=epoch)
        logger.add_scalar(tag='train/psnr', scalar_value=(train_psnr /len(train_loader)), global_step=epoch)
        logger.log_info(message)
        with torch.no_grad():
            model.eval()
            valid_vq_loss = 0.0
            valid_rec_loss = 0.0
            valid_spec_loss = 0.0
            valid_psnr = 0.0
            valid_st = 0.0
            valid_st_std = 0.0
            for x,smr,maskenerge in tqdm(valid_loader):
                x = x.to(args.device)
                smr=smr.to(args.device)
                maskenerge=maskenerge.to(args.device)
                x_wav= get_input(x)
                G_x, vq_loss= model(x_wav,is_quan_on_val)
                rec_loss,psnr,st,st_std= criterion(
                            x_wav,
                            G_x,
                            args=args)
                #spec_loss=torch.mean(stft(x_wav,G_x))
                spec_loss=fft_loss(x,G_x,smr)
                vq_loss=vq_loss
                valid_spec_loss+=spec_loss.item()
                valid_rec_loss += rec_loss.item()
                valid_vq_loss += vq_loss.item()
                valid_psnr+=psnr.item()
                valid_st+=st.item()
                valid_st_std+=st_std.item()
            if True:
                latest_model = model.state_dict()
                if valid_psnr > best_val_loss:
                    best_val_loss = valid_psnr
                    best_val_epoch = epoch
                    latest_save = {}
                    latest_save['latest_model'] = latest_model
                    latest_save['epoch'] = epoch
                    latest_save['optimizer_g'] = optimizer_g.state_dict()
                    latest_save['lr_scheduler_g'] = lr_scheduler_g.state_dict()
                    if args.log:
                        torch.save(latest_save, args.PATH + '/latest.pth')
                    count=0
                else:
                    count+=1            
            message = '<epoch:{:d}, <recon_loss_valid:{:.6f},st_valid:{:.5f}, spec_loss_valid:{:.5f},psnr_valid:{:.5f},best_epoch:{:d}>'.format(
                epoch, valid_rec_loss /len(valid_loader),valid_st /len(valid_loader),valid_spec_loss /len(valid_loader),valid_psnr /len(valid_loader),best_val_epoch)
            logger.add_scalar(tag='valid/st-sim', scalar_value=(valid_st /len(valid_loader)), global_step=epoch)
            logger.add_scalar(tag='valid/psnr', scalar_value=(valid_psnr /len(valid_loader)), global_step=epoch)
            logger.add_scalar(tag='valid/spec_loss', scalar_value=(valid_spec_loss /len(valid_loader)), global_step=epoch)
            logger.add_scalar(tag='valid/st-std', scalar_value=(valid_st_std/len(valid_loader)), global_step=epoch)    
            logger.log_info(message)
            if count>=30:
                message='over patience early stop !'
                logger.log_info(message)
                break
        if is_quan_on_val:
            lr_scheduler_q.step()
        else:
            lr_scheduler_g.step()

if __name__ == '__main__':
    main()
