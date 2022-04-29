import torch
import sys
sys.path.append('options')
from test_options import TestOptions
sys.path.append('data')
from data_loader import CreateDataLoader
sys.path.append('model')
from model_Loader import CreateModel
 
opt = TestOptions().parse()

data_loader = CreateDataLoader(opt)
model = CreateModel(opt)

total_steps = 0
for i, data in enumerate(data_loader):
    total_steps += len(data)
    model.forward(data)
    print(i)
    model.save_result()
