#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
from util import cal_loss, IOStream
from data import ModelNet40
import tqdm
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import visdom
import sklearn.metrics as metrics
from model_octree_capsule_cls_v8c import OctreeCaps
#from model_octree_capsule_cls_v4ori import OctreeCaps
import re

def find_largest_number_in_filenames(directory, word, extension):
    largest_number = None
    pattern = r'{}_(\d+)\{}'.format(word, extension)
    for filename in os.listdir(directory):
        if filename.endswith(extension) and word in filename:
            match = re.search(pattern, filename)
            if match:
                number = int(match.group(1))
                if largest_number is None or number > largest_number:
                    largest_number = number
    return largest_number

def losshistory(log,epoch,name):
    #'checkpoints/%s/models/model.pt' % args.exp_name
    csv_file_path = f"checkpoints/{args.exp_name}/models/{name}_log_{epoch}.csv"

    # Create a DataFrame with an "iteration" column and "loss" column

    df = pd.DataFrame(log)

    if os.path.isfile(csv_file_path):
        # The file exists, so you can load it
        loaded_df = pd.read_csv(csv_file_path)

        # Concatenate the new data with the loaded data, preserving the index
        df = pd.concat([loaded_df, df], ignore_index=True)
        df.to_csv(csv_file_path, index=False)  # Save to the same file
    else:
        # The file doesn't exist, so save your DataFrame to the CSV file
        df.to_csv(csv_file_path, index=False)

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py checkpoints'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points,treedepth=args.treedepth), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'octreecaps':
        model = OctreeCaps(args=args).to(device)
    else:
        raise Exception("Not implemented")


    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    #scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)


    criterion = cal_loss
    best_test_acc = 0
    fold_results = {'train_loss': [], 'train_accuracy': [], 'Avg_Train_Accuracy':[]}
    test_results = { 'test_loss': [], 'test_accuracy': [], 'Avg_Test_Accuracy': []}

    directory_path = "checkpoints/{}/models/".format(args.exp_name)
    word_to_search = "model"
    file_extension = ".pt"
    largest_number = find_largest_number_in_filenames(directory_path, word_to_search, file_extension)
    init_epoch = 0

    if largest_number is not None:
        print("last epoch: ", largest_number)
        init_epoch = largest_number+1
        checkpoint = torch.load(f"checkpoints/{args.exp_name}/models/model_{largest_number}.pt")
        model.load_state_dict(checkpoint)
        print(f"loaded model_{largest_number}.pt")
    else:
        print("No train files matching the criteria found.\n start init")

    for epoch in range(init_epoch, args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        train_loss_temp = 0

        for idx , (data, label) in enumerate( tqdm.tqdm(train_loader)):

            # print(f'data shape = {data.shape}')# batch 1024 3
            data, label = data.to(device), label.to(device).squeeze()
            # print(label.shape)

            # print(f'data shape = {data.shape}')
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)

            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size



            print(f'logit = {preds}\n,label ={label}\n ')
            print(f'train loss is = {loss.item()}, avg = {train_loss/count},acc= {torch.numel(torch.where(label==preds)[0])/torch.numel(label)}')

            if loss.item() < 0: break

            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_loss_temp = train_loss * 1.0 / count

        #if  train_loss_temp <1.5 :
        #    scheduler = CosineAnnealingLR(opt, args.epochs - init_epoch, eta_min=1e-10)
        #    args.scheduler = 'cos'
        if args.scheduler == 'cos' :

            scheduler.step()

        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-9:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-9:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-4
        print(f'current lr: {scheduler.get_last_lr()}')
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        print(f'train ture ={train_true},train ={train_pred}')
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss_temp ,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)


        ####################
        # Test
        ####################
        test_loss = 0.0
        count_test = 0.0
        model.eval()
        test_pred = []
        test_true = []
        test_loss_temp = 0
        for data, label in tqdm.tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()

            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count_test += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            print(f'logit = {preds},\nlabel ={label}\n,count ={count_test} ')
            print(f'test loss is = {loss.item()}, avg = {test_loss / count_test},acc= {torch.numel(torch.where(label == preds)[0]) / torch.numel(label)}')

        test_loss_temp=test_loss*1.0/count_test
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc_test = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss_temp,
                                                                              test_acc,
                                                                              avg_per_class_acc_test)
        io.cprint(outstr)
        if test_acc >= best_test_acc or epoch%9==0:
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
            torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/model_{epoch}.pt' )




        fold_results['train_loss'].append(train_loss_temp )
        fold_results['train_accuracy'].append( train_acc)
        fold_results['Avg_Train_Accuracy'].append(avg_per_class_acc)

        test_results['test_loss'].append(test_loss_temp)
        test_results['test_accuracy'].append(test_acc)
        test_results['Avg_Test_Accuracy'].append(avg_per_class_acc_test)




        vis.scatter( torch.tensor([epoch]),torch.tensor([train_loss_temp ]), win='Train Loss', update='append',
                 name=f'Epoch_{epoch}', opts={'title': 'Train Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})
        vis.scatter(torch.tensor([epoch]), torch.tensor([test_loss_temp]), win='Test Loss', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})

        vis.scatter( torch.tensor([epoch]),torch.tensor([train_acc]), win='Train Accuracy', update='append',
                 name=f'Epoch_{epoch}', opts={'title': 'Train Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})
        vis.scatter(torch.tensor([epoch]), torch.tensor([test_acc]), win='Test Accuracy', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})


        vis.scatter(torch.tensor([epoch]), torch.tensor([avg_per_class_acc]), win='Test  Accuracy',
                    update='append',name=f'Epoch_{epoch}', opts={'title': 'Avg Train Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})
        vis.scatter(torch.tensor([epoch]), torch.tensor([avg_per_class_acc_test]), win='Test  Accuracy', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Avg Test Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})

        losshistory(fold_results,epoch,'train')
        losshistory(test_results, epoch,'test')

#=======
##eval##
#=======

def LoadAllModelFile(directory, word, extension):
    largest_number = None
    pattern = r'{}_(\d+)\{}'.format(word, extension)
    modelfilellist =[]
    epochlist =[]

    for filename in os.listdir(directory):

        if filename.endswith(extension) and word in filename:
            match = re.search(pattern, filename)

            if match:
                modelfilellist.append(directory+match.group())
                epochlist.append(int(match.group(1)))
    return modelfilellist, epochlist

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'octreecaps':
        model = OctreeCaps(args=args).to(device)
    else:
        raise Exception("Not implemented")
    #Try to load models



    directory_path = "checkpoints/{}/models/".format(args.exp_name)
    word_to_search = "model"
    file_extension = ".pt"
    modellist,epochlist = LoadAllModelFile(directory_path, word_to_search, file_extension)

    for idx ,modelfile in enumerate(modellist):
        model = OctreeCaps(args=args).to(device)
        model = nn.DataParallel(model)
        try:

           model.load_state_dict(torch.load(modelfile))
        except Exception as e :
            print(e)
            print(f'not this model,skip this {modelfile}')
            continue
        model = model.eval()
        test_acc = 0.0
        count = 0.0
        test_true = []
        test_pred = []
        criterion = cal_loss
        test_loss =0
        test_results = {'test_loss': [], 'test_accuracy': [], 'Avg_Test_Accuracy': []}

        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()

            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            print(
                f'test loss is = {loss.item()}, avg = {test_loss / count},acc= {torch.numel(torch.where(label == preds)[0]) / torch.numel(label)}')
        test_loss_temp = test_loss * 1.0 / count
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
        test_results['test_loss'].append(test_loss_temp)
        test_results['test_accuracy'].append(test_acc)
        test_results['Avg_Test_Accuracy'].append(avg_per_class_acc)
        epoch = epochlist[idx]
        losshistory(test_results, epoch, 'test_eval')


        vis.scatter(torch.tensor([epoch]), torch.tensor([test_loss_temp]), win='Test Loss', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})


        vis.scatter(torch.tensor([epoch]), torch.tensor([test_acc]), win='Test Accuracy', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})

        vis.scatter(torch.tensor([epoch]), torch.tensor([avg_per_class_acc]), win='Test  Accuracy',
                    update='append', name=f'Epoch_{epoch}',
                    opts={'title': 'Avg Train Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})







        io.cprint(outstr)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='octreecaps', metavar='N',
                        choices=['pointnet', 'dgcnn','dgccaps','octreecaps'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--scheduler', type=str, default='step', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
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
    parser.add_argument('--eval', type=bool,  default=False,
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

    _init_()
    vis = visdom.Visdom(env=args.exp_name)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args,io)


