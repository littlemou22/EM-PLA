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

# seed = 100#100#3407
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = torch.device('cuda')

def set_gpu(data, device):
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))
    return data_gpu

def c_index(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair is not 0:
        return summ / pair
    else:
        return 0

def metrics_reg(targets, predicts):
    mae = metrics.mean_absolute_error(y_true=targets, y_pred=predicts)
    rmse = metrics.mean_squared_error(y_true=targets, y_pred=predicts, squared=False)
    r = scipy.stats.mstats.pearsonr(targets, predicts)[0]

    x = [[item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x, y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5
    ci = c_index(targets, predicts)

    return [mae, rmse, r, sd, ci]


def my_test(test_loader, metadata, filepath, model_mode):
    p_affinity = []
    y_affinity = []
    print(filepath)
    m_state_dict = torch.load(filepath)
    best_model = LBAPredictor(metadata, model_mode).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = best_model(data)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())
            # print(data[0].y.cpu().tolist())
    
    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)
    return p_affinity, y_affinity, affinity_err[0], affinity_err[1], affinity_err[2], affinity_err[3] , affinity_err[4] 

def test(test_set_list):
    for test_set in test_set_list:
        print(test_set.len)
        metadata = test_set[0][1].metadata()

        batch_vars = ["atom_coords", "seq"]
        batch_size = 16
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, follow_batch=batch_vars, shuffle=False)
        filepath = './simple/'#best_model #ablitation_model

        # filepath_list = ['./best_model/simple_capla_noRelu_100_metrics_log_0.0001_0.1_10_best_model.pt'
        #     './ablitation_model/simple_capla_noEnv_noRelu_100_metrics_log_0.0001_0.1_10_best_model.pt']
        for filename in os.listdir(filepath):
            
            if filename.endswith('.pt') and 'simple_capla_noRelu_100_metrics_log_1e-05_0.1_3_best_model' in filename:
                print(filename)
                # 获取文件名（去除.pt后缀）
                model_mode = filename[:-3]#+'_validation'
                path = filepath+filename
                # try:
                p_aff, y_aff, test_mae, test_rmse, test_r, test_sd, test_ci = my_test(test_loader, metadata, path, model_mode)
                print(model_mode)
                print(f'最佳模型的测试集指标, R: {test_r}, RMSE: {test_rmse}, MAE: {test_mae}, SD: {test_sd}, CI: {test_ci}')
                print(f"{test_r:.3f} {test_rmse:.3f} {test_mae:.3f} {test_sd:.3f} {test_ci:.3f}")

                # except:
                #     print()

if __name__ == '__main__':
    print("Loading dataset")
    test_set_list = []
    test_set1 = PLBA_Dataset('file', './data/core-set-2013.pkl') #core-set-2013 #core-set-2013_95 #core-set #core-set_262
    # test_set2 = PLBA_Dataset('file', './data/core-set-2013_95.pkl') #core-set-2013 #core-set-2013_95 #core-set #core-set_262
    test_set3 = PLBA_Dataset('file', './data/core-set_init5.pkl') #core-set-2013 #core-set-2013_95 #core-set #core-set_262
    # test_set4 = PLBA_Dataset('file', './data/core-set_262.pkl') #core-set-2013 #core-set-2013_95 #core-set #core-set_262
    test_set5 = PLBA_Dataset('file', './data/CSAR-HIQ_36.pkl')
    test_set6 = PLBA_Dataset('file', './data/CSAR-HIQ_51.pkl')
    # test_set7 = PLBA_Dataset('file', './data/csar_87.pkl')
    # test_set_list = [test_set1, test_set2, test_set3, test_set4, test_set5, test_set6, test_set7]
    # test_set_list = [test_set1, test_set3, test_set5, test_set6]
    test_set_list = [test_set3]
    test(test_set_list)