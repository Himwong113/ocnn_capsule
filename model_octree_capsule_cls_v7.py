import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from collections import OrderedDict


class OctreeKeyConv(nn.Module):
    def __init__(self, args, channel, layer=6):
        super(OctreeKeyConv, self).__init__()
        self.channel = channel
        self.layer = layer
        """
        self.bn1 = nn.BatchNorm1d(self.channel)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv1 = nn.Sequential(nn.Conv1d(1, self.channel, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   self.dp1)
        """
        """
        self.bn1 = nn.BatchNorm3d(self.channel)  # 64
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv1 = nn.Sequential(
            nn.Conv3d(args.treedepth + 1, self.channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1),
                      bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.02),
            self.dp1,
            nn.AvgPool3d(kernel_size=(2, 2, 2)))

        self.bn2 = nn.BatchNorm3d(self.channel)  # 32
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.channel, self.channel, kernel_size=(3, 3, 3), padding=(0, 1, 1), stride=(3, 1, 1),
                      bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.02),
            self.dp2)  # ,
        # nn.MaxPool3d(kernel_size=(1, 2, 2)))
        """

        """
        self.bn3 = nn.BatchNorm3d(self.channel)#16
        self.dp3 = nn.Dropout(p=args.dropout)
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.channel, self.channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
            self.dp3,
            nn.MaxPool3d(kernel_size=(1, 2, 2)))

        
        self.bn4 = nn.BatchNorm3d(self.channel)  #8
        self.dp4 = nn.Dropout(p=args.dropout)
        self.conv4 = nn.Sequential(
            nn.Conv3d(self.channel, self.channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
            self.dp4,
            nn.MaxPool3d(kernel_size=(1, 2, 2)))
        
        """

        self.conv = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', nn.Conv3d(args.treedepth + 1, self.channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1),bias=False)),
                ('bn3', nn.BatchNorm3d(self.channel)),
                ('dp',nn.Dropout(p=args.dropout)),
                ('mp1', nn.AvgPool3d(kernel_size=(2,2,2))),
            ]))
            for _ in range(self.layer)])


        #print(f'conv={self.conv}')
        self.bn5 = nn.BatchNorm3d(self.channel)  # 8
        self.dp5 = nn.Dropout(p=args.dropout)
        self.conv5 = nn.Sequential(
            nn.Conv3d(self.channel*self.layer, self.channel, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.02),
            self.dp5,
        )

    def forward(self, x):
        # print(x.shape) #(B,1,4681)
        x = [depth(x) for depth in self.conv]

        x = torch.concat(x,dim=1)


        #x = self.conv2(x)
        # print(f'conv2={x.shape}')
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.conv5(x)
        # print(f'conv5={x.shape}')
        # print(x1.shape) #(B,12,937)
        # x2 = self.conv2(x1)
        # print(x2.shape) #(B,12,234)
        # x3 = self.conv3(x2)
        # print(x3.shape) #(B,12,116)
        #
        ##print(x.shape) #(B,12,937+234+116)=(B,12,1287)
        #
        # x= self.conv4(x3)#(B,12,1287)
        # print(x.shape)
        #
        #
        # x =torch.permute(x,(0,2,1))

        return x


class LatentCapsLayer(nn.Module):
    def __init__(self, prim_caps_size=1287, prim_vec_size=40, latent_caps_size=40, latent_vec_size=40):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01 * torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            # print(f'c_ij ={c_ij}')
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)


class OctreeCaps(nn.Module):
    def __init__(self, args, channel=30):
        super(OctreeCaps, self).__init__()
        self.channel = channel
        self.args = args
        self.layer = 6

        self.treedepth = args.treedepth
        self.noderepeatcount = []
        for i in range(args.treedepth + 1):
            self.noderepeatcount.append(int(math.pow(8, args.treedepth) / math.pow(8, i)))

        # print(self.noderepeatcount)
        self.OctreeKeyConv = OctreeKeyConv(channel=self.channel, args=args,layer=self.layer)

        self.Latentcaps = LatentCapsLayer(prim_caps_size=int((2**(args.treedepth-1))**3),
                                          prim_vec_size=self.channel,
                                          )
        """
        self.Latentcaps_ver2 = LatentCapsLayer(
                                                prim_caps_size=int( self.nodesum/4),
                                                prim_vec_size= self.channel,
                                                latent_caps_size=40,
                                                latent_vec_size=40)
        """

        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.Linear_cls = nn.Linear(256,1)
        # self.bn1 = nn.BatchNorm1d(40)
        # self.dp1 = nn.Dropout(args.dropout)
        # self.Linear_cls2 = nn.Linear(128, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.dp2 = nn.Dropout(args.dropout)
        # self.Linear_cls3 = nn.Linear(128, output_channel)
        # self.W_fea = nn.Parameter(0.01*torch.rand (40,output_channel))

    def forward(self, data):

        """
         #data = data.permute(0, 2, 1)  # batch  4681 1
        #Latent = self.Latentcaps_ver2 (data)
        encoder = self.OctreeKeyConv_ver2(data)  # batch 12 1287
        result = torch.sum(encoder, dim=-1)
        result = F.softmax(result,dim=-1)

        """
        batch_size = data.shape[0]
        temp = 0
        repeatvector = []
        for i in range(self.treedepth + 1):
            #print(f'{i}:')
            #print(temp)
            #print(int(self.noderepeatcount[int(-i - 1)]))
            repeatvector.append(data[:, :, temp:temp + self.noderepeatcount[int(-i - 1)]])
            # print(data[:,:,temp:temp+self.noderepeatcount[int(-i-1)]].shape)
            temp = temp + int(self.noderepeatcount[int(-i - 1)])
            repeatvector[i] = repeatvector[i].repeat(1, 1, self.noderepeatcount[int(i)])
            #print(repeatvector[i].shape)


        repeatvector = torch.concat(repeatvector, dim=1).to('cuda' if not self.args.no_cuda else 'cpu')
        #print(f'self.noderepeatcount[0]={math.ceil(self.noderepeatcount[0] ** (1 / 3))}')
        a = math.ceil(self.noderepeatcount[0] ** (1 / 3))
        repeatvector = repeatvector.view(batch_size, self.treedepth + 1, a, a, a)

        #print(repeatvector.shape)
        dwrepeatvector = self.OctreeKeyConv(repeatvector)
        print(f'dwrepeatvector.shape ={dwrepeatvector.shape}')

        Latent = self.Latentcaps(dwrepeatvector.view(batch_size, int((2**(self.treedepth-1))**3), self.channel))
        result = torch.sum(Latent, dim=-1)

        # print(f'result ={result.shape}')

        """
        encoder = self.OctreeKeyConv(data)  # batch 12 1287
        print(f'encoder ={encoder.shape}')
        Latent = self.Latentcaps_ver2(encoder)
        print(f'latent = {Latent.shape}')  # batch ,40,40
        result = torch.sum(Latent, dim=-1)
        #print(f'result ={result}')
        print(f'result ={result.shape}') #batch ,40
        """

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='octreecaps', metavar='N',
                        choices=['pointnet', 'dgcnn', 'dgccaps', 'octreecaps'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--treedepth', type=int, default=4, metavar='N',
                        help='treedepth_l')
    args = parser.parse_args()
    numpoint = 0
    for i in range(args.treedepth + 1):
        numpoint += int(math.pow(8, i))
    x = torch.rand((args.batch_size, 1, numpoint)).to('cuda')
    model = OctreeCaps(args).to('cuda')
    out = model(x)
    print(out)
