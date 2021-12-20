import matplotlib.pyplot as plt

from util import *
from trainer import Trainer
from net import gtnet
import time
import pandas as pd
from train_multi_step import args
import seaborn as sns

def main(runid):
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # predefined_A = load_adj(args.adj_data)
    predefined_A = torch.ones(args.num_nodes, args.num_nodes)
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler,
                     device, args.cl)
    # change model
    expid = 1
    if expid == 0:
        load_path = "data/graph/normal"
    elif expid == 1:
        load_path = "data/graph/adv"
    else:
        exit(0)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(expid) + "_" + str(runid) + ".pth"))
    print("load model successfully.")
    # test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)

    for iter, (x1, x2, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx1 = torch.Tensor(x1).to(device)
        testx1 = testx1.transpose(1, 3)
        testx2 = torch.Tensor(x2).to(device)
        testx2 = testx2.transpose(1, 3)
        with torch.no_grad():
            preds, adp, adp_prev, KLloss = engine.model(testx1, testx2)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        # pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, :, i]
        metrics = metric(yhat[:, i, :, :], real, adp, adp_prev)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    save_heatmap(dataloader, engine, load_path)
    exit(0)
    return mae, mape, rmse

def save_heatmap(dataloader, engine, path):
    device = torch.device(args.device)
    # data = pd.read_json(path_or_buf="data/MARLActionPrediction/MARLActionPrediction/gamedata/" + str(0) + ".json", orient=dict)["actions"]
    data = pd.read_json(path_or_buf="data/game_data_adv/" + str(0) + ".json", orient=dict)["actions"]
    for iter, (x1, x2, y) in enumerate(dataloader['temp_adv_loader'].get_iterator()):
        testx1 = torch.Tensor(x1).to(device)
        testx1 = testx1.transpose(1, 3)
        testx2 = torch.Tensor(x2).to(device)
        testx2 = testx2.transpose(1, 3)
        with torch.no_grad():
            preds, adp, adp_prev, KLloss = engine.model(testx1, testx2)
        print(adp.shape)
        for i in range(len(data)):
            ax = sns.heatmap(adp[i].cpu())
            plt.show()
            figure = ax.get_figure()
            figure.savefig(path + "_adv" + "/" + "sns_heatmap_" + str(i) + ".jpg")  # 保存图片
        # print(adp)
        # print(len(data))


if __name__ == "__main__":
    mae = []
    mape = []
    rmse = []
    adv_mae = []
    adv_mape = []
    adv_rmse = []
    t1 = time.time()
    for i in range(args.runs):
        m1, m2, m3,  = main(i)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        # adv_mae.append(adv_m1)
        # adv_mape.append(adv_m2)
        # adv_rmse.append(adv_m3)
    # test
    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)


    t2 = time.time()
    print("time cost:", t2 - t1)
    # test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [0, 1, 2]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))


    '''
    # adv test data
    outputs = []
    realy = torch.Tensor(dataloader['y_adv_test']).to(device)
    realy = realy.transpose(1, 3)

    for iter, (x1, x2, y) in enumerate(dataloader['adv_test_loader'].get_iterator()):
        testx1 = torch.Tensor(x1).to(device)
        testx1 = testx1.transpose(1, 3)
        testx2 = torch.Tensor(x2).to(device)
        testx2 = testx2.transpose(1, 3)
        with torch.no_grad():
            preds, adp, adp_prev, KLloss = engine.model(testx1, testx2)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    adv_mae = []
    adv_mape = []
    adv_rmse = []
    for i in range(args.seq_out_len):
        # pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, :, i]
        metrics = metric(yhat[:, i, :, :], real, adp, adp_prev)
        log = 'Evaluate best model on adv test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        adv_mae.append(metrics[0])
        adv_mape.append(metrics[1])
        adv_rmse.append(metrics[2])
    '''
    '''
    # adv_test
    adv_mae = np.array(adv_mae)
    adv_mape = np.array(adv_mape)
    adv_rmse = np.array(adv_rmse)

    adv_amae = np.mean(adv_mae, 0)
    adv_amape = np.mean(adv_mape, 0)
    adv_armse = np.mean(adv_rmse, 0)

    adv_smae = np.std(adv_mae, 0)
    adv_smape = np.std(adv_mape, 0)
    adv_srmse = np.std(adv_rmse, 0)
    '''
    '''
    # adv_test data
    print('adv_test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [0, 1, 2]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, adv_amae[i], adv_armse[i], adv_amape[i], adv_smae[i], adv_smape[i], adv_srmse[i]))
    '''
