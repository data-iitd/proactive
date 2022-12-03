import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
import Utils
from process import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
import pdb

def prepare_dataloader(opt):
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            num_goals = data['dim_goals']
            data = data[dict_name]
            return data, int(num_types), int(num_goals)

    print('Loading All Datasets...')
    train_data, num_types, num_goals = load_data(opt.data + 'train.pkl', 'train')
    test_data, _, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types, num_goals

def train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt):
    model.train()

    total_event_ll = 0
    total_time_se = 0
    total_event_rate = 0
    total_goal_rate = 0
    total_num_event = 0
    total_num_pred = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        # Likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # Type Prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)
        
        # Time Prediction
        se = Utils.time_loss(prediction[1], event_time)

        # Goal Prediction
        goal_loss, pred_num_goal = Utils.goal_loss(prediction[2], event_goal, pred_loss_goal)

        # Scales to stabilize training
        scale_time_loss = 1
        scale_goal_loss = 10
        loss = event_loss + pred_loss + goal_loss/scale_goal_loss + se / scale_time_loss
        loss.backward()

        optimizer.step()

        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_goal_rate += pred_num_goal.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_se / total_num_pred
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_num_pred, mae

def eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt):
    model.eval()

    total_event_ll = 0
    total_time_se = 0
    total_event_rate = 0
    total_goal_rate = 0
    total_num_event = 0
    total_num_pred = 0
    total_seqs = 0

    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2, desc='  - (Validation) ', leave=False):
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

            enc_out, prediction = model(event_type, event_time)

            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            pred_goal, seq_num = Utils.pred_goal(prediction[2], event_goal)
            se = Utils.time_loss(prediction[1], event_time)

            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_goal_rate += pred_goal.item()
            total_seqs += seq_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_se / (total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_seqs, mae

def train(model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
    test_acc_list = []
    test_goal_list = []
    test_mae_list = []
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_goal, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt)
        print('(Training) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, Itv. GPA: {goal: 8.5f}'.format(type=train_type, mae=train_time, goal=train_goal))

        start = time.time()
        test_event, test_type, test_goal, test_time = eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt)
        print('(Testing) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, GPA: {goal: 8.5f}'.format(type=test_type, mae=test_time, goal=test_goal))

        test_acc_list += [test_type]
        test_goal_list += [test_goal]
        test_mae_list += [test_time]
        print('Best ACC: {pred: 8.5f}, MAE: {mae: 8.5f}, GPA: {gpa: 8.5f}'.format(pred=max(test_acc_list), mae=min(test_mae_list), gpa=max(test_goal_list)))

        scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=32)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    opt = parser.parse_args()

    opt.device = torch.device('cuda')
    trainloader, testloader, num_types, num_goals = prepare_dataloader(opt)

    model = Transformer(
        num_types=num_types,
        num_goals=num_goals,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        pred_loss_goal = Utils.LabelSmoothingLoss(opt.smooth, num_goals, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        pred_loss_goal = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)

if __name__ == '__main__':
    main()
