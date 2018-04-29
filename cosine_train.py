from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.triplet_image_loader import TripletImageLoader
from utils.openface import prepareOpenFace
from tripletnet import Tripletnet
from visdom import Visdom
import numpy as np
import dlib
from PIL import Image

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TripletNet trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--TPI', type=int, default=100, metavar='TPI',
                    help='triplets per individual, only used when random sampling (default: 100)')
parser.add_argument('--workers', type=int, default=32, metavar='w',
                    help='number of prefetch processes (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for testing (default: 8)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=.3, metavar='M',
                    help='margin for triplet loss (default: 0.3)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='cosine_training', type=str,
                    help='name of experiment')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Use cuda: ", args.cuda)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter 
    plotter = VisdomLinePlotter(args.name)

    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader(
            'name_thumbPaths_train.csv', 
            transform=transforms.Compose([
                transforms.Resize(96),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
            ]), 
            triplets_per_individual = args.TPI
        ),
        batch_size=args.batch_size, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader(
            'name_thumbPaths_train.csv', 
            transform=transforms.Compose([
                transforms.Resize(96),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
            ]), 
            triplets_per_individual = 100
        ),
        batch_size=args.batch_size, num_workers=args.workers)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.features = models.squeezenet1_1(pretrained=True).features
            self.embedding = nn.Sequential(
                nn.Linear(2048, 512),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(512, 128)
            )

        def forward(self, x):
            x = self.features(x)
            x = nn.functional.adaptive_max_pool2d(x, 2)
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            return self.embedding(x)
            
            
    net = Net()
    if args.cuda:
        net.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    criterion = torch.nn.CosineEmbeddingLoss(margin=args.margin)    
    
    optimizer = optim.Adam(net.parameters(), lr=5e-5, weight_decay=1e-5)

    n_parameters = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch)
        # torch.cuda.empty_cache()
        # evaluate on validation set
        acc = test(test_loader, net, criterion, epoch)
        
        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

def train(train_loader, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    net.train()
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        if args.cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)

        # compute POSITIVE output
        embedded_anchor = net(anchor)
        embedded_positive = net(positive)
        positive_target = torch.autograd.Variable(torch.ones(len(embedded_anchor)).cuda())
        pos_loss = criterion(embedded_anchor, embedded_positive, positive_target)
        
        # compute gradient for correcting positive distances
        #optimizer.zero_grad()
        #pos_loss.backward(retain_graph=True)
        #optimizer.step()
        
        
        # compute NEGATIVE output
        embedded_negative = net(negative)
        negative_target = torch.autograd.Variable(-1 * torch.ones(len(embedded_anchor)).cuda())
        neg_loss = criterion(embedded_anchor, embedded_negative, negative_target)
        
        # compute gradient for correcting negative distances
        #optimizer.zero_grad()
        #neg_loss.backward()
        #optimizer.step()

        loss_triplet = pos_loss + neg_loss
        loss_norm = embedded_anchor.norm(2) + embedded_positive.norm(2) + embedded_negative.norm(2)
        
        loss = loss_triplet + 1e-4 * loss_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # measure accuracy and record loss
        acc = accuracy(embedded_anchor, embedded_positive, embedded_negative, margin=args.margin)
        losses.update(loss_triplet.data[0], anchor.size(0))
        accs.update(acc, anchor.size(0))
        emb_norms.update(loss_norm.data[0]/3, anchor.size(0))

        
        

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
            # log avg values to somewhere
            plotter.plot('acc_train', 'train', epoch * len(train_loader) + batch_idx, accs.avg)
            plotter.plot('loss_train', 'train', epoch * len(train_loader) + batch_idx, losses.avg)
            plotter.plot('emb_norms_train', 'train', epoch * len(train_loader) + batch_idx, emb_norms.avg)
    

def test(test_loader, net, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to evaluation mode
    net.eval()
    for batch_idx, (anchor, positive, negative) in enumerate(test_loader):
        if args.cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor = Variable(anchor, volatile=True)
        positive = Variable(positive, volatile=True)
        negative = Variable(negative, volatile=True)

        # compute POSITIVE output
        embedded_anchor = net(anchor)
        embedded_positive = net(positive)
        positive_target = torch.autograd.Variable(torch.ones(len(embedded_anchor)).cuda())
        pos_loss = criterion(embedded_anchor, embedded_positive, positive_target)
                
        
        # compute NEGATIVE output
        embedded_negative = net(negative)
        negative_target = torch.autograd.Variable(-1 * torch.ones(len(embedded_anchor)).cuda())
        neg_loss = criterion(embedded_anchor, embedded_negative, negative_target)
        

        loss_triplet = pos_loss + neg_loss
        loss_norm = embedded_anchor.norm(2) + embedded_positive.norm(2) + embedded_negative.norm(2)
        loss = loss_triplet + 1e-4 * loss_norm

        # measure accuracy and record loss
        acc = accuracy(embedded_anchor, embedded_positive, embedded_negative, margin=args.margin)
        losses.update(loss_triplet.data[0], anchor.size(0))
        accs.update(acc, anchor.size(0))
        emb_norms.update(loss_norm.data[0]/3, anchor.size(0))     

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    plotter.plot('acc', 'test', epoch, accs.avg)
    plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='asdasd'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Iter',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(
                X=np.array([x]), 
                Y=np.array([y]), 
                env=self.env, win=self.plots[var_name], name=split_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def positive_accuracy(emb_anc, emb_pos, margin=.5):
    sim = torch.nn.CosineSimilarity(dim=1)
    correct_similarity = (sim(emb_anc, emb_pos) > margin).sum()
    
    acc = correct_similarity.float() / len(emb_anc)
    return acc.data[0]


def total_accuracy(emb_anc, emb_pos, emb_neg, margin=.5):
    sim = torch.nn.CosineSimilarity(dim=1)
    correct_similarity = (sim(emb_anc, emb_pos) > margin)
    correct_dissimilarity = (sim(emb_anc, emb_neg) < margin)
    
    total_score = correct_similarity.sum() + correct_dissimilarity.sum()
    acc = total_score.float() / 2 / len(emb_anc)
    #print(correct_similarity.data[0], correct_dissimilarity.data[0])
    return acc.data[0]


def triplet_accuracy(emb_anc, emb_pos, emb_neg, margin=.5):
    sim = torch.nn.CosineSimilarity(dim=1)
    correct_similarity = (sim(emb_anc, emb_pos) > margin)
    correct_dissimilarity = (sim(emb_anc, emb_neg) < margin)

    triplet_score = torch.mul(correct_similarity, correct_dissimilarity).sum()
    acc = triplet_score.float() / len(emb_anc)
    return acc.data[0]


def accuracy(emb_anc, emb_pos, emb_neg, margin=.5):
    # just a wrapper
    #return positive_accuracy(emb_anc, emb_pos, margin=margin)
    #return total_accuracy(emb_anc, emb_pos, emb_neg, margin=margin)
    return triplet_accuracy(emb_anc, emb_pos, emb_neg, margin=margin)

if __name__ == '__main__':
    main()    
