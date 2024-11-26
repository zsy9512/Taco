'''
Author: shengyu
Date: 2024-04-19 09:23:16
LastEditors: shengyu
LastEditTime: 2024-07-09 10:22:11
Description: dataset functions
'''
import os
import glob
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing
#from src.phm.PsychohapticModel import PsychohapticModel as PhModel
from pathlib import Path
class PWQDataset(Dataset):
    """Dataset to load PWQ data."""

    def __init__(self, file_dir,max_len=2800):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(file_dir + "/*.wav"))
        _, self.sr = torchaudio.load(self.filenames[0])
        self.max_len =max_len # 24000
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        ans[:, :audio.shape[1]] = audio
        return ans
class PWQDataset_SMR(Dataset):
    """Dataset to load PWQ data."""

    def __init__(self, file_dir,max_len=2800):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(file_dir + "/*.wav"))
        _, self.sr = torchaudio.load(self.filenames[0])
        self.max_len =max_len # 24000
    def __len__(self):
        return len(self.filenames)
    def get_dict(self):
        return self.dict
    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        ans[:, :audio.shape[1]] = audio
        # pmres=self.phm.getSMR(ans.reshape(-1,))
        # smr=torch.Tensor(pmres.SMR).reshape(1,-1)
        path=self.filenames[index].split(os.sep)
        path[7]="smr"
        filename= os.sep.join(path)
        smrfile=filename.replace(".wav", ".pt")
        smrfile=torch.load(smrfile,weights_only=True)
        smr,maskenerge=smrfile[0],smrfile[1]
        return ans,smr.reshape(-1,1),maskenerge.reshape(-1,1)