import  torch
import  numpy as np

from torchvision import models
import advertorch.attacks as attacks
import torch.nn.functional as F
import random

attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD", "Single_pixel", "DeepFool"]

def setAttack(str_at, net, eps, args):
    e = eps/255.
    iter = args.iter
    clip_max = 1.0
    clip_min = -1.0
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "FGSM":
        return attacks.GradientSignAttack(net, eps=e, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "BIM_L2":
        return attacks.L2BasicIterativeAttack(net, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "BIM_Linf":
        return attacks.LinfBasicIterativeAttack(net, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "MI-FGSM":
        return attacks.MomentumIterativeAttack(net, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min) # 0.3, 40
    elif str_at == "CnW":
        return attacks.CarliniWagnerL2Attack(net, args.n_way, max_iterations=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "EAD":
        return attacks.ElasticNetL1Attack(net, args.n_way, max_iterations=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "DDN":
        return attacks.DDNL2Attack(net, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "Single_pixel":
        return attacks.SinglePixelAttack(net, max_pixels=iter, clip_max=clip_max, clip_min=clip_min)
    elif str_at == "DeepFool":
        return attacks.DeepfoolLinfAttack(net, args.n_way, eps=e, nb_iter=iter, clip_max=clip_max, clip_min=clip_min)
    else:
        print("wrong type Attack")
        exit()

def setModel(str, n_way, imgsz, imgc, pretrained='IMAGENET1K_V1'):
    str = str.lower()
    if str=="resnet18":
        model = models.resnet18(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet34":
        model = models.resnet34(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet50":
        model = models.resnet50(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet101":
        model = models.resnet101(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet152":
        model = models.resnet152(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="densenet121":
        model = models.densenet121(weights=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        return model
    elif str=="mobilenet_v2":
        model = models.mobilenet_v2(weights=pretrained)
        num_ftrs = model.classifier._modules["1"].in_features
        model.classifier._modules["1"] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(imgc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    elif str=="alexnet":
        model = models.alexnet()
        feature_in = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(feature_in, n_way)
        return model
    else:
        print("wrong model")
        print("possible models : resnet18, resnet34, resnet50, resnet101, resnet152, alexnet, densenet121, mobilenet_v2")
        exit()

def set_seed(seed=706):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict(x, y, model):
    logit = model(x)
    pred = F.softmax(logit, dim=1)
    outputs = torch.argmax(pred, dim=1)
    correct_count = (outputs == y).sum().item()
    loss = F.cross_entropy(logit, y)

    return correct_count, loss

def adv_predict(args, at_type, at_bound, x, y, model, mode='train'):
    at = setAttack(at_type, model, at_bound, args)
    advx = at.perturb(x, y)
    correct_count, adv_loss = predict(advx, y, model)

    if mode=='train':
        return adv_loss
    else:
        return correct_count, adv_loss

