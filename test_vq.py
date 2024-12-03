'''
Author: shengyu
Date: 2024-05-14 15:18:45
LastEditors: shengyu
LastEditTime: 2024-07-17 16:28:28
Description: 请填写简介
'''
'''
python /home/zsy/workplace/model/vq_test.py \
    --resume_path=/home/zsy/workplace/model/output/checkpoints/2024-08-13-19-56vq_4/latest.pth --nq 4 \
    --bins 256
'''
import argparse
import os
import sys
import math
import typing as tp
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
import torch
from src.modules.net import TouchStream
from tqdm import tqdm
from src.utils.dataset import PWQDataset_SMR , PWQDataset
from src.utils.loss import loss,criterion,Wfft_loss,fft_loss
from src.utils.utils import Logger
from src.utils.utils import seed_everything
from src.quantization.amcode import *
from tqdm import tqdm
#from vector_quantize_pytorch import ResidualVQ,VectorQuantize,GroupedResidualVQ
from src.quantization import ResidualVectorQuantizer,GroupVectorQuantizer
from vibromaf.metrics import stsim,spqi
'''
python /home/zsy/workplace/taco/test_vq.py \
    --resume_path=/home/zsy/workplace/taco/output/checkpoints/nq_4_256/latest.pth --nq 4 \
    --bins 256\
    --group_num 2
'''
datadict=[]
def get_parser():
    parser =argparse.ArgumentParser(
        'encodec',
        description='vib compression and decompression'
        'do a compression/decompression cycle for a npy file.')
    parser.add_argument(
        '--input',
        type=Path,
        default='./dataset/pwq_or/test',
        help='Input file.')
    parser.add_argument(
        '--output',
        type=Path,
        default='./output_001/Ger',
        nargs='?',
        help='Output file.')
    parser.add_argument(
        '--resume_path', type=str, default='resume_path', help='resume_path')
    parser.add_argument(
        '-r',
        '--rescale',
        action='store_true',
        help='Automatically rescale the output to avoid clipping.')
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
    return parser
def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)
def rescale(data):
    limit = 1.0
    mx = np.max(np.abs(data))
    if rescale:
        data = data * min(limit / mx, 1)
    else:
        data = data.clamp(-limit, limit)
    return data

def test_one(args, file_root,datadict, model):
    #load data from file_root
    import torchaudio
    #data=torch.from_numpy(np.load(file_root).astype(np.float32).reshape(1,1,-1)).cuda()
    data = torchaudio.load(file_root)[0].cuda().unsqueeze(0)
    wav = data.to(memory_format=torch.contiguous_format).float()
    # quantize,正常流程编码解码
    #quantized= model.encode(wav)
    #wavout = model.decode(quantized)
    #这里先直接拿结果吧，一样的
    quantized=np.random.randint(0,(args.bins-1),size = [int(args.nq*int(np.log2(args.bins))),1])
    out, _=    model(data)
    result={"data":wav,"quantized":quantized,"out":out}
    datadict.append(result)
def amc_trans(qunatized):
    data=qunatized.tobytes()
    data_size = len(data)
    
    #print("Calculating probability of {data_size} bytes..")
    # 统计字节频数
    #pro_dic, keys, accum_pro = cal_pr(data)
    pro_dic=cal_prob(data)
    amcode_ls = ""
    C_upls = []
    C_downls = []
    byte_num=700
    integra = math.ceil(data_size / byte_num)  # 迭代次数
    #byte_num =math.ceil(data_size / integra)  # 每次编码的字节数
    #print(integra,byte_num)
    # 编码
    for k in range(integra):
        #C_up, C_down, amcode = encode(data[byte_num * k: byte_num * (k + 1)], pro_dic, data_size)
        amcode=arithEncode(data[byte_num * k: byte_num * (k + 1)], pro_dic)
        amcode_ls += amcode
        #C_upls.append(C_up)
        #C_downls.append(C_down) 
    #codebyte_num =math.ceil(len(amcode_ls) / 8)
    #解码
    #decodebyte_ls = []
    # for k in range(integra):
    #     if (k == integra - 1) & (data_size % byte_num != 0):
    #         decodebyte_ls += decode(C_upls[k], C_downls[k], pro_dic, keys, accum_pro, data_size % byte_num,
    #                                 data_size)
    #     else:
    #         decodebyte_ls += decode(C_upls[k], C_downls[k], pro_dic, keys, accum_pro, byte_num, data_size)
    # errornum=0
    # for j in range(data_size):
    #     if data[j] != decodebyte_ls[j]:
    #         errornum += 1
    # print("Error rate: %.3f%%" % (errornum / data_size * 100))  # 误码率
    return len(amcode_ls)
    #t_end = datetime.now()      
def evaluations(args,datadict):
    results=[]
    plist=[]
    slist=[]
    cramlist=[]
    crlist=[]
    stlist=[]
    splist=[]
    for re in datadict:
        ref =re["data"].detach().cpu().squeeze(0).squeeze(0).numpy()
        quantized=re["quantized"].astype(np.uint16)
        deg=re["out"].detach().cpu().squeeze(0).squeeze(0).numpy()
        #print(ref.shape,quantized.shape,deg.shape)
        deg=rescale(deg)
        diff =(ref - deg)**2
        psnr=10*np.log10((4)/(np.mean(diff)))
        snr=10*np.log10(np.sum(ref**2)/np.sum(diff))
        st,ss=stsim.st_sim(deg,ref,eta=0.5)
        sp=spqi.spqi(deg,ref)
        ##amc
        codenum=amc_trans(quantized)
        cr_am=(512*16)/codenum
        cr=(512*16)/(args.nq*np.log2(args.bins)*32)
        plist.append(psnr)
        slist.append(snr)
        cramlist.append(cr_am)
        crlist.append(cr)
        stlist.append(st)
        splist.append(ss)
        results.append({"psnr":psnr,"snr":snr,"CR_am":cr_am,"CR":cr,"ST":st,"SP":sp})
    print("psnr: ",np.mean(np.array(plist))," snr: ",np.mean(np.array(slist)),
          " CR: ",np.mean(np.array(crlist))," CR_am: ",np.mean(np.array(cramlist)),
          " st: ",np.mean(np.array(stlist))," sp: ",np.mean(np.array(splist)))
    return results
def test_batch():
    datadict=[]
    args = get_parser().parse_args()
    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")
    input_lists = os.listdir(args.input)
    input_lists.sort()
    bins=args.bins
    n_q=args.nq
    n_group=args.group_num
    if args.group_num>1 and n_q>1:#使用分组量化
        quantizer =  GroupVectorQuantizer(dimension=128, codebook_dim=128,n_q=n_q, bins=bins ,n_codes_group=n_group)
    else:
        quantizer = ResidualVectorQuantizer(dimension=128, codebook_dim=128,n_q=n_q, bins=bins)
    model= TouchStream(input_dim=128, quantizer=quantizer)
    parameter_dict = torch.load(args.resume_path)['latest_model']
    #parameter_dict = torch.load(args.resume_path)['model']
    model.load_state_dict(parameter_dict)  # load model weights
    model.cuda()
    model.eval()
    os.makedirs(args.output, exist_ok=True)
    for vib in tqdm(input_lists):
        test_one(args=args,file_root=os.path.join(args.input, vib),datadict=datadict,model=model)
    res=evaluations(args,datadict)
    pd.DataFrame(res).to_json(os.path.join(args.output, 'phm_vq_'+str(n_q)+"_"+str(bins)+'.json'))
if __name__ == '__main__':
    test_batch()
