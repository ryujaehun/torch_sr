import torch.utils.data as data
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
import torch
import h5py
from copy import deepcopy as dp
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.target = hf.get("label")

    def __getitem__(self, index):
       # print(self.data[index,0,:,:].reshape(1,self.data.shape[2],self.data.shape[2]).shape)
       # print(self.target[index,0,:,:].reshape(1,self.target.shape[2],self.target.shape[2]).shape)
        return torch.from_numpy(self.data[index,0:1,:,:].reshape(1,self.data.shape[2],self.data.shape[2])).float(), torch.from_numpy(self.target[index,0:1,:,:].reshape(1,self.target.shape[2],self.target.shape[2])).float()
    def __len__(self):
        return self.data.shape[0]