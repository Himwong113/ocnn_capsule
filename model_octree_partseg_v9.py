import torch
import torch.nn as nn
import ocnn

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from attention_decoder import build_decoder
import math
import open3d as o3d


class OctreeKeyConv(nn.Module):
    def __init__(self, args, channel, lastlayer=8):
        super(OctreeKeyConv, self).__init__()
        self.channel = channel
        self.lastlayer = lastlayer
        """
        self.bn1 = nn.BatchNorm1d(self.channel)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv1 = nn.Sequential(nn.Conv1d(1, self.channel, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   self.dp1)
        """

        self.bn1 = nn.BatchNorm3d(self.channel)  # 64
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, self.channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
            self.dp1,
            nn.MaxPool3d(kernel_size=(1, 2, 2)))

        self.bn2 = nn.BatchNorm3d(self.channel)  # 32
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.channel, self.channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1),
                      bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
            self.dp2)  # ,
        # nn.MaxPool3d(kernel_size=(1, 2, 2)))
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

        self.bn5 = nn.BatchNorm3d(lastlayer)  # 8
        self.dp5 = nn.Dropout(p=args.dropout)
        self.conv5 = nn.Sequential(
            nn.Conv3d(self.channel, self.lastlayer, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
            self.dp5,
        )

    def forward(self, x):
        # print(x.shape) #(B,1,4681)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(f'conv2={x.shape}')
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.conv5(x)
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
    def __init__(self, prim_caps_size=1287, prim_vec_size=40, latent_caps_size=40, latent_vec_size=40,
                 num_iterations=11):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01 * torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
        self.num_iterations = num_iterations

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def forward(self, x):

        # W = F.softmax(self.W,dim=0)
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = self.num_iterations
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            # print(f'c_ij ={c_ij}')
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)

        return v_j.squeeze(-2)


class pointcloud_embed(nn.Module):
    def __init__(self, depth, dropout):
        super(pointcloud_embed, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.node_count = int(math.pow(8, self.depth))

        self.dp1b = nn.Dropout(p=self.dropout)
        self.bn1b = nn.BatchNorm1d(int(self.node_count))
        self.pointcloud_conv = nn.Sequential(nn.Conv1d(3, int(self.node_count), kernel_size=1, bias=False),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             self.bn1b,
                                             self.dp1b)
        """
        self.dp1b = nn.Dropout(p=self.dropout)
        self.bn1b = nn.BatchNorm1d(int(self.node_count / 8))
        self.pointcloud_conv = nn.Sequential(nn.Conv1d(3, int(self.node_count / 8), kernel_size=1, bias=False),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             self.bn1b,
                                             self.dp1b)

        self.dp1bb = nn.Dropout(p=self.dropout)
        self.bn1bb = nn.BatchNorm1d(int(self.node_count / 4))
        self.pointcloud_conv2 = nn.Sequential(
            nn.Conv1d(int(self.node_count / 8), int(self.node_count / 4), kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.02),
            self.bn1bb,
            self.dp1bb)

        self.dp1bbb = nn.Dropout(p=self.dropout)
        self.bn1bbb = nn.BatchNorm1d(int(self.node_count))
        self.pointcloud_conv3 = nn.Sequential(
            nn.Conv1d(int(self.node_count / 4), int(self.node_count), kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.02),
            self.bn1bbb,
            self.dp1bbb)
        """

    def forward(self, pointcloud):
        pointcloud = torch.permute(pointcloud, (0, 2, 1))
        pointcloud = self.pointcloud_conv(pointcloud)
        # pointcloud = self.pointcloud_conv2(pointcloud)
        # pointcloud = self.pointcloud_conv3(pointcloud)
        print(f'pointcloud ={pointcloud.shape}')  # batch 4096,point

        return pointcloud


class OctreeCaps_partseg(nn.Module):
    def __init__(self, args, output_channel=60,
                 seg_num=50,
                 octree_channel=40,
                 latent_vec_size=32,
                 label_channel=64,
                 pointcloud_channel=128):
        super(OctreeCaps_partseg, self).__init__()
        self.args = args
        self.channel = output_channel
        self.octree_channel = octree_channel
        self.latent_vec_size = latent_vec_size
        self.num_points = args.num_points
        self.depth = args.treedepth
        self.label_channel = label_channel
        self.pointcloud_channel = pointcloud_channel
        self.node_count = int(math.pow(8, self.depth))
        self.cnnlastlayer = 8
        self.treedepth = args.treedepth
        self.noderepeatcount = []
        for i in range(args.treedepth + 1):
            self.noderepeatcount.append(int(math.pow(8, args.treedepth) / math.pow(8, i)))

        self.OctreeKeyConv = OctreeKeyConv(channel=self.octree_channel, args=args, lastlayer=self.cnnlastlayer)

        self.dp1a = nn.Dropout(p=args.dropout)
        self.bn1a = nn.BatchNorm1d(self.label_channel)
        self.label_embedd = nn.Sequential(nn.Conv1d(16, self.label_channel, kernel_size=1, bias=False),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          self.bn1a,
                                          self.dp1a)

        self.Latentcaps = LatentCapsLayer(prim_caps_size=1024,
                                          prim_vec_size=self.cnnlastlayer * (self.treedepth + 1),
                                          )

        self.pointcloud_embconv = pointcloud_embed(depth=self.depth, dropout=args.dropout)

        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(seg_num)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.dp3 = nn.Dropout(p=args.dropout)
        self.dp4 = nn.Dropout(p=args.dropout)
        self.dp5 = nn.Dropout(p=args.dropout)
        self.dp6 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Sequential(nn.Conv1d(1088, 1024, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.02), self.dp2)
        self.conv3 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.02), self.dp3)
        self.conv4 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.02), self.dp4)
        self.conv5 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.02), self.dp5)

        self.conv6 = nn.Sequential(nn.Conv1d(128, seg_num, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.02), self.dp6)

    def forward(self, data, pointcloud, l):

        batch_size = data.size(0)
        temp = 0
        repeatvector = []
        for i in range(self.treedepth + 1):
            # print(temp)
            # print(int(self.noderepeatcount[int(-i-1)]))
            repeatvector.append(data[:, :, temp:temp + self.noderepeatcount[int(-i - 1)]])
            # print(data[:,:,temp:temp+self.noderepeatcount[int(-i-1)]].shape)
            temp = temp + int(self.noderepeatcount[int(-i - 1)])
            repeatvector[i] = repeatvector[i].repeat(1, 1, self.noderepeatcount[int(i)])
            # print(repeatvector[i].shape)

        repeatvector = torch.concat(repeatvector, dim=1).to('cuda' if not self.args.no_cuda else 'cpu')
        # print(f'repeatvector ={repeatvector.shape}')
        repeatvector = repeatvector.view(batch_size, 1, self.treedepth + 1,
                                         int(math.pow(self.noderepeatcount[0], 1 / 2)),
                                         int(math.pow(self.noderepeatcount[0], 1 / 2)))

        # print(f'repeatvector ={repeatvector.shape}')
        dwrepeatvector = self.OctreeKeyConv(repeatvector)
        dwrepeatvector = dwrepeatvector.view(batch_size, 1024, self.cnnlastlayer * (self.treedepth + 1))

        # print(f'dwrepeatvector ={dwrepeatvector.shape}')
        Latent = self.Latentcaps(dwrepeatvector)

        # print(f'latent = {Latent.shape}')  # batch ,40,40

        result = torch.bmm(dwrepeatvector, Latent)
        result = result.max(dim=-1, keepdim=True)[0]
        # print(f'result shape ={result.shape}')

        # num_categoties =16
        # num_categoties =['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        # print(f'l={l.shape}')
        emb_l = self.label_embedd(
            l)  # (batch_size, num_categoties, 1) -> (batch_size, channel, latent_vec_size) #batch 32,64
        # print(f'emb_l={emb_l.shape}')
        emb_l = emb_l.max(dim=-1, keepdim=True)[0]  # batch 64 1
        # print(f'emb_l={emb_l.shape}')
        emb_vec = torch.concat([result, emb_l], dim=1)
        # print(f'emb_vec1 ={emb_vec.shape}')

        emb_vec = emb_vec.repeat(1, 1, int(math.pow(8, self.depth)))  # batch 1024+16 4096
        # print(f'emb_vec2 ={emb_vec.shape}')
        pointcloud = self.pointcloud_embconv(pointcloud)
        result = torch.bmm(emb_vec, pointcloud)
        ## MLP part
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        result = self.conv6(result)

        # print(f' result = {result.shape}')#batch 50 4096
        """
        pointcloud =torch.permute(pointcloud,(0,2,1))
        pointcloud = self.pointcloud_conv(pointcloud)
        pointcloud = self.pointcloud_conv2(pointcloud)
        pointcloud = self.pointcloud_conv3(pointcloud)
        print(f'pointcloud ={pointcloud.shape}') #batch 4096,point
        """

        # print(f' result = {result.shape}') #(batch, numpoint, part seg count)

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
    parser.add_argument('--batch_size', type=int, default=5, metavar='batch_size',
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
    parser.add_argument('--num_points', type=int, default=2048,
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

    x = torch.rand(args.batch_size, 1, 4681).to('cuda')
    model = OctreeCaps_partseg(args, output_channel=16).to('cuda')
    l = torch.rand(args.batch_size, 16).to('cuda')

    # from function_test_octreeindex import create_zero_tensor

    pcd = torch.randint(0, 2, (args.batch_size, args.num_points, 3)).to('cuda').float()
    print(pcd)
    out = model(x, pcd, l)
    print(out.shape)

