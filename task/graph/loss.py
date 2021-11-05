import torch
import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, recon, target):
        return self.loss(recon, target)


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target, num_classes):
        return self.loss(logit, target)


class LossPredLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_loss, target_loss):
        pred_loss = (pred_loss - pred_loss.flip(0))[:len(pred_loss)//2]
        
        target_loss = (target_loss - target_loss.flip(0))[:len(target_loss)//2]
        target_loss = target_loss.detach()

        one = 2 * torch.sign(torch.clamp(target_loss, min=0)) - 1

        loss = torch.sum(torch.clamp(1.0 - one * pred_loss, min=0))
        loss = loss / pred_loss.size(0)

        return loss


class RankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, pred_loss, target_loss):
        target = (target_loss - target_loss.flip(0))[:target_loss.size(0) // 2]
        target = target.detach()
        ones = torch.sign(torch.clamp(target, min=0))

        pred_loss = (pred_loss - pred_loss.flip(0))[:pred_loss.size(0) // 2]
        pred_loss = torch.sigmoid(pred_loss)

        return self.bce(pred_loss, ones)


class GDistanceLoss(nn.Module):
    def __init__(self):
        super(GDistanceLoss, self).__init__()

        self.loss = nn.MSELoss()

    def forward(self, features, loss):
        target1 = torch.sqrt(loss)
        target1 = target1.detach()
        distance1 = torch.sqrt(torch.sum(torch.pow(features, 2), dim=1))

        target2 = torch.sqrt((loss * loss.flip(0))[:loss.size(0) // 2])
        target2 = target2.detach()
        distance2 = torch.sqrt(torch.sum(torch.pow((features - features.flip(0))[:features.size(0) // 2], 2), dim=1))

        return self.loss(distance1, target1) + self.loss(distance2, target2)