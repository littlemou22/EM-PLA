import os
import torch
import warnings
from sklearn import metrics

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.linear_model import LinearRegression
from easydict import EasyDict
import scipy
import yaml

from models.lba_model import LBAPredictor
from binding_data import PLBA_Dataset
# from dmasif_encoder.data_iteration import iterate_surface_precompute
from tqdm import tqdm
# def iterate_surface_precompute(dataloader):
#     processed_dataset = []
#     for it, data in enumerate(dataloader):
#         processed_dataset.append(data)
#     return processed_dataset


device = torch.device('cuda')


def set_gpu(data, device):
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))
    return data_gpu


def metrics_reg(targets, predicts):
    mae = metrics.mean_absolute_error(y_true=targets, y_pred=predicts)
    rmse = metrics.mean_squared_error(y_true=targets, y_pred=predicts, squared=False)
    r = scipy.stats.mstats.pearsonr(targets, predicts)[0]

    x = [[item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x, y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

    return [mae, rmse, r, sd]

def my_val(model, val_loader, device, scheduler):
    p_affinity = []
    y_affinity = []

    model.eval()
    loss_epoch = 0
    n = 0
    for data in val_loader:
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = model(data)
            loss = F.mse_loss(predict, data[0].y)
            loss_epoch += loss.item()
            n += 1

            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())

    scheduler.step(loss_epoch / n)

    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)

    return affinity_err


def my_test(test_loader, metadata, kf_filepath, lr, decay_factor, decay_paient, model_mode):
    p_affinity = []
    y_affinity = []

    m_state_dict = torch.load(kf_filepath + f"/{model_mode}_metrics_log_{lr}_{decay_factor}_{decay_paient}" + '_best_model.pt')
    best_model = LBAPredictor(metadata, model_mode).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = best_model(data)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())

    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)

    f_log = open(file=(kf_filepath + f"/{model_mode}_metrics_log_{lr}_{decay_factor}_{decay_paient}.txt"), mode="a")
    str_log = 'test_mae: ' + str(affinity_err[0]) + ' test_rmse: ' + str(affinity_err[1]) + ' test_r: ' + str(
        affinity_err[2]) + ' test_sd: ' + str(affinity_err[3]) + '\n'
    f_log.write(str_log)
    f_log.close()

    return affinity_err[0], affinity_err[1]

def my_train(train_loader, val_loader, test_loader, kf_filepath, model, lr, decay_factor, decay_paient, model_mode):
    print('start training')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=decay_factor, patience=decay_paient, verbose=True)

    loss_list = []
    best_mae = float('inf')
    best_rmse = float('inf')
    for epoch in range(100):
        model.train()
        loss_epoch = 0
        n = 0
        total_batches = len(train_loader)  
        for data in train_loader:
            data = set_gpu(data, device)
            optimizer.zero_grad()
            out = model(data)

            loss = F.mse_loss(out, data[0].y)
            loss_epoch += loss.item()

            if n % (total_batches // 3) == 0: 
                print(f'epoch: {epoch}, i: {n}, loss: {loss.item()}')

            loss.backward()
            optimizer.step()
            n += 1
        loss_list.append(loss_epoch / n)

        print('epoch:', epoch, ' loss:', loss_epoch / n)

        val_err = my_val(model, val_loader, device, scheduler)
        val_mae = val_err[0]
        val_rmse = val_err[1]

        print(val_rmse)

        if val_rmse < best_rmse:# and val_mae < best_mae:
            print('********save model*********')
            torch.save(model.state_dict(), kf_filepath + f"/{model_mode}_metrics_log_{lr}_{decay_factor}_{decay_paient}" + '_best_model.pt')
            best_mae = val_mae
            best_rmse = val_rmse

            f_log = open(file=(kf_filepath + f"/{model_mode}_metrics_log_{lr}_{decay_factor}_{decay_paient}.txt"), mode="a")
            str_log = 'epoch: ' + str(epoch) + ' val_mae: ' + str(val_mae) + ' val_rmse: ' + str(val_rmse) + '\n'
            f_log.write(str_log)
            f_log.close()

            test_mae, test_rmse = my_test(test_loader, metadata, kf_filepath, lr, decay_factor, decay_paient, model_mode)
            print(f'最佳模型的验证集指标, MAE: {val_mae}, RMSE: {val_rmse}')
            print(f'最佳模型的测试集指标, MAE: {test_mae}, RMSE: {test_rmse}')
        print("---------------------------------------------------")
    print(f'训练结束，最佳RMSE为{test_rmse}')
    # plt.plot(loss_list)
    # plt.ylabel('Loss')
    # plt.xlabel("time")
    # plt.savefig(kf_filepath + '/loss.png')
    # plt.show()
# hid_dim=64
# out_dim=64
# num_layers=2
# egnn_layers=1
if __name__ == '__main__':

    seed = 100#100#3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("Loading dataset")
    train_set = PLBA_Dataset('file', './data/general-set.pkl')#general
    val_set = PLBA_Dataset('file', './data/refined-set.pkl')#refined
    test_set = PLBA_Dataset('file', './data/core-set.pkl') #_3graph
    metadata = train_set[0][1].metadata()

    batch_vars = ["atom_coords", "seq"]
    batch_size = 16
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=False)
    filepath = './simple/'

    lr = 1e-4
    mode_1 = ['simple']
    mode_2 = ['capla_noRelu(2)']#capla_hasRelu
    factor_list = [0.1]
    paient_list = [10]
    for m1 in mode_1:
        for m2 in mode_2:
            for factor in factor_list:
                for paient in paient_list:
                    if((factor==0.1 and paient==10) or (factor==0.5 and paient==3)):
                        mode = f'{m1}_{m2}_{seed}'
                        print(mode)
                        model = LBAPredictor(metadata, mode).to(device)
                        my_train(train_loader, val_loader, test_loader, filepath, model, lr, factor, paient, 
                                model_mode=mode)