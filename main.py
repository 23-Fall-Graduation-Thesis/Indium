import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import argparse
from model.models import Conv4
from dataset_dir.datasets import datasetload
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='Dataset type')
    parser.add_argument('--model', default='Conv4', help='Model type')
    parser.add_argument('--mode', type=str, default='pre', help='Pretrain(pre) or Finetune(fine)')
    parser.add_argument('--finetune', default='custom', help='Finetuning type')
    
    parser.add_argument('--epoch', type=int, default=50, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--step', type=str2bool, nargs='?', const=True, default=False, help='StepLR scheduler')
    parser.add_argument('--stepsize', type=int, default=50, help='StepLR schedular step_size')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR scheduler gamma value')
    parser.add_argument('--wd', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--device', type=int, default=1, help='CUDA device')

    return parser

def select_model(model_name):
    if model_name == 'Conv4':
        model = Conv4()
    # elif model_name == 'Conv8':
    #     model = Conv8()
    # elif model_name == 'Conv12':
    #     model = Conv12()
    else:
        raise ValueError(f'Undefined Model.')
    
    return model

def train(model, criterion, optimizer, trainloader):
    model.train()
    
    train_loss = 0
    train_acc = 0
    for data, target in trainloader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        train_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        criterion.backward()
        optimizer.step()
    
    train_loss = train_loss / len(trainloader.dataset)
    train_acc = train_acc / len(trainloader.dataset) * 100
    
    return train_loss, train_acc

def validation(model, criterion, validloader):
    model.eval()
    
    valid_loss = 0
    valid_acc = 0
    with torch.no_grad():
        for data, target in validloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            valid_acc += pred.eq(target.view_as(pred)).sum().item()
    
    valid_loss = valid_loss / len(validloader.dataset)
    valid_acc = valid_acc / len(validloader.dataset) * 100
    
    return valid_loss, valid_acc
    
def test(model, criterion, testloader, checkpt):
    model.load_state_dict(torch.load(checkpt))
    model.eval()
    
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_acc += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = test_loss / len(testloader.dataset)
    test_acc = test_acc / len(testloader.dataset) * 100
    
    return test_loss, test_acc

if __name__ == 'main':
    print("엥")
    # arguments parsing
    args = arg_parse(argparse.ArgumentParser())
    
    print("잉")
    # random seed
    np.random.seed(2023)
    torch.manual_seed(2023)
    
    # cuda device
    device = torch.device("cuda:" + str(args.device))
    
    # dataset load
    trainloader, validloader, testloader = datasetload()
    
    # experiment setting values
    if args.step == True:
        setting = "epoch"+str(args.epoch)+"_lr:"+str(args.lr)+"_stepsize:"+str(args.stepsize) + "_gamma:"+str(args.gamma)+"_wd:"+str(args.wd)
    else:
        setting = "epoch"+str(args.epoch)+"lr:"+str(args.lr)+"_wd:"+str(args.wd)
    
    # checkpoint file path & model define
    if args.mode == 'pre':
        checkpt = "./model/weight/pretrain/"+str(args.model)+"/"+str(args.dataset)+"_"+setting+".pt"
        board_name = str(args.model)+"/"+str(args.dataset)+"_"+setting
        writer = SummaryWriter("./results/log/pretrain/"+board_name)
        model = select_model(args.model).to(device)
    # elif args.mode == 'fine'
    #     if args.finetune == 'custom':
    #         checkpt = "./model/weight/finetune/custum/"+str(args.model)+"/"+str(args.dataset)+"_"+setting+".pt"
    #         board_name = str(args.model)+"/"+str(args.dataset)+"_"+setting
    #         writer = SummaryWriter("./results/log/finetune/custom/"+board_name)
    #         model = custom().to(device)
    #     elif args.finetune == 'optimal':
    #         checkpt = "./model/weight/finetune/optimal/"+str(args.model)+"/"+str(args.dataset)+"_"+setting+".pt"
    #         board_name = str(args.model)+"/"+str(args.dataset)+"_"+setting
    #         writer = SummaryWriter("./results/log/finetune/optimal"+board_name)
    #         model = optimal().to(device)
    #     else:
    #         raise ValueError(f'Invalid finetune mode input.')
    # else:
    #    raise ValueError(f'Invalid mode input')
    
    print('model: ', args.model, ' dataset:', args.dataset, 'mode')
    print('Experiment Setting: ', setting)
    
    # loss, optimizer define
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.step == True:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    print('Start training')
    best = 99999999
    best_epoch = 0
    bad_count = 0
    for epoch in range(args.epoch):
        train_loss, train_acc = train(model, criterion, optimizer, trainloader, device)
        valid_loss, valid_acc = validation(model, criterion, validloader, device)
        if (epoch + 1) % 5 == 0:
            print('Epoch:{:04d}'.format(epoch+1), 'train loss:{:.3f}'.format(train_loss), 'acc:{:.2f}'.format(train_acc))
            print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
        if valid_loss < best:
            best = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpt)
            bad_count = 0
        else:
            bad_count += 1
        
        if bad_count == 10:
            break
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('Acc/val', valid_acc, epoch)
    print('Finish')
    print('start test')
    
    test_loss, test_acc = test(model, criterion, testloader, checkpt, device)
    print('Load {}th epoch'.format(best_epoch))
    print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))