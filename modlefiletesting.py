import os
import re


def LoadAllModelFile(directory, word, extension):
    largest_number = None
    pattern = r'{}_(\d+)\{}'.format(word, extension)
    modelfilellist =[]
    for filename in os.listdir(directory):

        if filename.endswith(extension) and word in filename:
            match = re.search(pattern, filename)

            if match:
                modelfilellist.append(directory+match.group())
    return modelfilellist

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='octreecaps_cls_v4_c30', metavar='N',
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

    directory_path = "checkpoints/{}/models/".format(args.exp_name)
    word_to_search = "model"
    file_extension = ".pt"
    modellist = LoadAllModelFile(directory_path, word_to_search, file_extension)
    print(modellist)