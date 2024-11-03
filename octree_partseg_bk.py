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
class OctreeKeyConv(nn.Module):
    def __init__(self,args,channel):
        super( OctreeKeyConv,self).__init__()
        self.channel =channel
        self.bn1 = nn.BatchNorm1d(self.channel)
        self.bn2 = nn.BatchNorm1d(self.channel)
        self.bn3 = nn.BatchNorm1d(self.channel)
        self.bn4 = nn.BatchNorm1d(self.channel)


        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.dp3 = nn.Dropout(p=args.dropout)
        self.dp4 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Sequential(nn.Conv1d(1, self.channel, kernel_size=5,padding=2,stride=5, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   self.dp1)
        self.conv2 = nn.Sequential(nn.Conv1d(self.channel, self.channel, kernel_size=5, padding=0, stride=4, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   self.dp2)
        self.conv3 = nn.Sequential(nn.Conv1d(self.channel, self.channel, kernel_size=4, padding=0, stride=2, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   self.dp3)

        self.linear1 = nn.Sequential(nn.Conv1d(self.channel, self.channel, kernel_size=1, padding=0, stride=1, bias=False),
                           self.bn4,
                           nn.LeakyReLU(negative_slope=0.2),
                            self.dp4)#no drop out in cuurent test



    def forward(self, x):
        #print(x.shape) #(B,1,4681)
        x1 = self.conv1(x)
        #print(x1.shape) #(B,12,937)
        x2 = self.conv2(x1)
        #print(x2.shape) #(B,12,234)
        x3 = self.conv3(x2)
        #print(x3.shape) #(B,12,116)
        x = torch.concat([x1,x2,x3],dim=2)
        #print(x.shape) #(B,12,937+234+116)=(B,12,1287)

        x= self.linear1(x)#(B,12,1287)


        x =torch.permute(x,(0,2,1))

        return x

class LatentCapsLayer(nn.Module):
    def __init__(self,  prim_caps_size=1287, prim_vec_size=40,latent_caps_size=40, latent_vec_size=40):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
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
            #print(f'c_ij ={c_ij}')
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)




class OctreeCaps_partseg(nn.Module):
    def __init__(self,args,output_channel =32,seg_num_all=50,latent_vec_size =64,N=3,h=3):
        super(OctreeCaps_partseg,self).__init__()
        self.channel = output_channel
        self.latent_vec_size = latent_vec_size
        self.num_points =args.num_points


        self.h = h
        self.OctreeKeyConv = OctreeKeyConv(channel=self.latent_vec_size,args=args)
        self.Latentcaps = LatentCapsLayer(
                                          prim_vec_size=self.latent_vec_size,
                                          latent_caps_size=self.latent_vec_size,
                                          latent_vec_size=self.channel)
        self.dp1 = nn.Dropout(p=args.dropout)

        self.bn1 = nn.LayerNorm(1)
        self.Latentcaps_label = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2),self.dp1)

        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(seg_num_all)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.dp3 = nn.Dropout(p=args.dropout)
        self.dp4 = nn.Dropout(p=args.dropout)
        self.dp5 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Sequential(nn.Conv1d(1351, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2),self.dp2)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2),self.dp3)
        self.conv4 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2),self.dp4)
        self.conv5 = nn.Sequential(nn.Conv1d(128, seg_num_all, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2),self.dp5)



    def forward(self,data,l):


        batch_size = data.size(0)

        #print(data.shape)
        encoder = self.OctreeKeyConv(data)  # batch 1287,32
        #print(f'encoder ={encoder.shape}')
        Latent = self.Latentcaps(encoder)
        #print(f'latent = {Latent.shape}')  # batch ,40,40


        result = torch.bmm(encoder, Latent).max(dim=-1)[0].unsqueeze(dim=-1)
        #print(f'result={result.shape}')  #12, 40,1287


        # num_categoties =16
        # num_categoties =['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        #print(f'l={l.shape}')
        caps_l = self.Latentcaps_label(l)  # (batch_size, num_categoties, 1) -> (batch_size, channel, latent_vec_size) #batch 32,64
        #print(f'caps_l={caps_l.shape}')
        emb_vec =  torch.concat([result,caps_l],dim=1)
        #print(f'emb_vec ={emb_vec.shape}')

        emb_vec =  emb_vec.repeat(1,1,self.num_points)

        ## MLP part
        result = self.conv2(emb_vec)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        #print(f' result = {result.shape}') #(batch, numpoint, part seg count)
        return result











if __name__=="__main__":
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

    x =torch.rand(12,1,4681).to('cuda')
    model =OctreeCaps_partseg(args,output_channel=16).to('cuda')
    l = torch.rand(12,16).to('cuda')
    out = model(x,l)
    print(out.shape)

