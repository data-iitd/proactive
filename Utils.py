import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def compute_event(event, non_pad_mask):
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result

def compute_integral_biased(all_lambda, time, non_pad_mask):
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result

def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    num_samples = 100
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral

def log_likelihood(model, data, time, types):
    non_pad_mask = get_non_pad_mask(types).squeeze(2)
    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll

def type_loss(prediction, types, loss_func):
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]
    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num

def goal_loss(prediction, types, loss_func):
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)
    
    gamma =  np.full((loss.shape[0], loss.shape[1]), 0.99)
    for i in range(loss.shape[1]):
        gamma[:,i] = gamma[:,i] **i
    gamma = torch.from_numpy(gamma).to("cuda:0")
    loss = loss * gamma
    loss = torch.sum(loss)
    return loss, correct_num

def pred_goal(prediction, types):
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    pred_type = pred_type.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()

    preds = []
    trs = []
    for i in range(len(truth)):
        id_ = np.argwhere(truth[i] == -1)
        if len(id_) == 0:
            preds.append(pred_type[i][-1])
        else:
            preds.append(pred_type[i][id_[0][0]-1])
        trs.append(truth[i][0])

    correct_num = np.sum(preds == trs)
    correct_num = torch.tensor([correct_num])
    total_seqs = torch.tensor([truth.shape[0]-1])
    return correct_num, total_seqs

def time_loss(prediction, event_time):
    prediction.squeeze_(-1)
    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    diff =  true - prediction
    se = torch.sum(torch.abs(diff))
    return se

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        non_pad_mask = target.ne(self.ignore_index).float()
        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
