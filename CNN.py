import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        conv1 = nn.Conv2d(3, 6, 5, 1)   # 6@