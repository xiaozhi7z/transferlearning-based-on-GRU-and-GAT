import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import torch
torch.cuda.set_device(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from tensorboardX import SummaryWriter

import math
import utils
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import random
import copy

from dataset import *
from models import *
import warnings
warnings.filterwarnings("ignore")


writer = SummaryWriter()

def cal_metrics(prediction_data, ground_truth):
    # print('prediction_result shape is', prediction_data.shape)
    # print('ground_truth shape is', ground_truth.shape)
    seq_len = prediction_data.shape[-1]
    MSE, RMSE, MAE, MAPE = [], [], [], []
    prediction_data = prediction_data.reshape((-1,seq_len))
    ground_truth = ground_truth.reshape((-1,seq_len))
    def cal_MAPE(pred, y):
        mask = y != 0
        diff = np.abs(np.array(y[mask]) - np.array(pred[mask]))
        return np.mean(diff / y[mask])

    for seq_index in range(seq_len):
        pred, y = prediction_data[:, seq_index], ground_truth[:, seq_index]
        MSE.append(mean_squared_error(pred, y))
        RMSE.append(mean_squared_error(pred, y) ** 0.5)
        MAE.append(mean_absolute_error(pred, y))
        MAPE.append(cal_MAPE(pred, y))

    results = {'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE}
    return results


def format_print(results):
    print('{:^40}'.format('PM2.5(t1-t6)'))
    metrics = list(results.keys())
    # print('results.keys:', metrics)   #{'MSE', 'RMSE', 'MAE', 'MAPE'}
    # print('results is',results)
    for metric in metrics:
        metric_value = results[metric]
        print(metric, end="\t")
        for i in range(len(metric_value)-1):
            print('{:<10}'.format("%.3f"%metric_value[i]), end=' ')
        print('{:<10}'.format("%.3f"%metric_value[i+1]))


def main(args):

    try:
        seed = 7
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Basic Info
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("INFO: device = ", device)
        print('now cuda num',torch.cuda.current_device())

        # Data for big source -----------------------------------------------------------------------------------
        city_num_source = 1
        data_path_source = r'/home/zzy6anl/cgy/GRU/data_fully/' + 'IAQI_fully_{}.csv'.format(city_num_source)
        # adjacency_matrix_path_source = r'/home/zzy6anl/cgy/DANN/edge/' + 'adj_mat_{}.npy'.format(city_num_source)
        
        train_x_source, train_y_source, val_x_source, val_y_source, test_x_source, test_y_source = get_data(data_path_source, 
                                                                history_len=args.history_len,
                                                                prediction_len=args.prediction_len, is_src= 1)
        
        print('train_dataset_source shape is', train_x_source.shape)
        print('val_dataset_source shape is', val_x_source.shape)
        print('test_dataset_source shape is', test_x_source.shape)
        # time_len,node_num,24,1

        # data for small target ---------------------------------------------------------------------------------
        city_num_target = 0
        data_path_target = r'/home/zzy6anl/cgy/GRU/data_fully/' + 'IAQI_fully_{}.csv'.format(city_num_target)
        train_x_target, train_y_target, val_x_target, val_y_target, test_x_target, test_y_target = get_data(data_path_target, 
                                                                history_len=args.history_len,
                                                                prediction_len=args.prediction_len, is_src= 0)
        
        print('train_dataset_target shape is', train_x_target.shape)
        print('val_dataset_target shape is', val_x_target.shape)
        print('test_dataset_target shape is', test_x_target.shape)

        # two city ------------------------------------------------------------------------------------------
        train_Dataset_combined = myDataset_combined(train_x_source, train_y_source, city_num_source,
                                                    train_x_target, train_y_target, city_num_target)
        train_dataloader_combined = DataLoader(train_Dataset_combined, batch_size=args.batch_size, shuffle=True, 
                                                drop_last =True)
        
        # 断点输出-----------------------------------------------------------------------------------------------
        # print('now 断点输出')
        # for step, (data_source, data_target) in enumerate(train_dataloader_combined):
        #     # print(data)
        #     data_source = data_source.to(device)
        #     data_target = data_target.to(device)
        #     # print('source x shape',data_source.x.shape)
        #     # print('target x shape',data_target.x.shape)
        #     # print('source edge', data_source.edge_index.shape)
        #     # print('target edge', data_target.edge_index.shape)
        #     # print('_target city num?',data_target.num_nodes)
        #     # print('_target city num?',data_source.num_nodes)
        #     print('_target data batch ',data_target.batch)
        #     print('_target data batch shape',data_target.batch.shape)
        #     print('_source data batch ',data_source.batch)
            
        # raise KeyboardInterrupt
        # x: 32, station_num, 24, 1 | y: 32, station_num, 6, 1 | edge_index: 2, edge_num*32
        
        # validation
        val_Dataset_combined = myDataset_combined(val_x_source, val_y_source, city_num_source,
                                                  val_x_target, val_y_target, city_num_target)
        val_dataloader_combined = DataLoader(val_Dataset_combined, batch_size=args.batch_size, shuffle=True, 
                                            drop_last =True)
        # test
        test_Dataset_combined = myDataset_combined(test_x_source, test_y_source, city_num_source,
                                                  test_x_target, test_y_target, city_num_target)
        test_dataloader_combined = DataLoader(test_Dataset_combined, batch_size=args.batch_size, shuffle=True, 
                                            drop_last =True)


        print('data have been ready')
        # Model
        model = mytry(args).to(device)

        # Optimizers --------------------------------------------------------------------------------------------
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=0, eps=1e-08)
        criterion = torch.nn.MSELoss()
        # criterion_featrue = toech.nn.Cr
        train_loss_list = np.zeros(args.epochs)
        val_loss_list = np.zeros(args.epochs)
        test_loss_list = np.zeros(args.epochs)
        print('========== Ready for training ==========')
        # lets start trainging !!! ------------------------------------------------------------------------------
        for epoch in range(args.epochs):
            model.train()
            start_time = time.time()
            total_loss = 0
            val_total_loss = 0
            test_total_loss = 0
            if epoch<args.epochs_pre:
                # print('still pre-train', epoch, 'epochs')
                # len_dataloader = len(train_dataloader)
                for step, (data_source, data_target) in enumerate(train_dataloader_combined):
                    # print('step',step,'data',data)
                    data_source = data_source.to(device)
                    data_target = data_target.to(device)
                    train_y_source = data_source.y.squeeze(-1)
                    train_y_traget = data_target.y.squeeze(-1)
                    
                    output_predict_source, output_predict_target, featrue_output_source, featrue_output_target, source_both_featrue, target_both_featrue= model(data_source, data_target, args.alpha)
                    # print('train_prediction_data shape is ',train_prediction_data.shape)
                    optimizer.zero_grad()
                    loss_predict_source = criterion(output_predict_source, train_y_source)
                    loss_predict_traget = criterion(output_predict_target, train_y_traget)
                    loss_featrue_source = criterion(featrue_output_source, source_both_featrue)
                    loss_featrue_traget = criterion(featrue_output_target, target_both_featrue)
                    loss_featrue_both = criterion(target_both_featrue, source_both_featrue)
                    writer.add_scalar('loss_predict_source', loss_predict_source, epoch)
                    writer.add_scalar('loss_predict_traget', loss_predict_traget, epoch)
                    writer.add_scalar('loss_featrue_source', args.loss_alpha*loss_featrue_source, epoch)
                    writer.add_scalar('loss_featrue_traget', args.loss_alpha*loss_featrue_traget, epoch)
                    writer.add_scalar('loss_featrue_both', loss_featrue_both, epoch)

                    loss = loss_predict_source + loss_predict_traget + args.loss_alpha*loss_featrue_source + args.loss_alpha*loss_featrue_traget + loss_featrue_both
                    writer.add_scalar('loss_total', loss, epoch)
                    loss.backward()
                    optimizer.step()
                    total_loss = total_loss + loss.item()


                avg_loss = total_loss / (step + 1)
                end_time = time.time()
                # print('prediction_data shape is', prediction_data.shape)
                # print('output_data shape is', output_data.shape)
                print("Epoch:", epoch, "/", args.epochs, "Training time is ", end_time - start_time, "avg loss is",
                    avg_loss)
                train_loss_list[epoch] = avg_loss

                # validation
                if epoch % 1 == 0:
                    model.eval()
                    with torch.no_grad():
                        for step_val, (data_source, data_target) in enumerate(val_dataloader_combined):
                            # print('test step is', step_test)
                            data_source = data_source.to(device)
                            data_target = data_target.to(device)
                            
                            val_y_source = data_source.y.squeeze(-1)
                            val_y_traget = data_target.y.squeeze(-1)
                            output_predict_source, output_predict_target, featrue_output_source, featrue_output_target, source_both_featrue, target_both_featrue= model(data_source, data_target, args.alpha)
                            # loss calculate
                            loss_predict_source = criterion(output_predict_source, val_y_source)
                            loss_predict_traget = criterion(output_predict_target, val_y_traget)
                            loss_featrue_source = criterion(featrue_output_source, source_both_featrue)
                            loss_featrue_traget = criterion(featrue_output_target, target_both_featrue)
                            loss_featrue_both = criterion(target_both_featrue, source_both_featrue)
                            val_loss = loss_predict_source + loss_predict_traget 
                            val_total_loss = val_total_loss + val_loss.item()
                            if step_val == 0:
                                val_prediction_result = output_predict_source
                                val_ground_truth_result = val_y_source
                            else:
                                val_prediction_result = torch.cat((val_prediction_result, output_predict_source), dim=0)
                                val_ground_truth_result = torch.cat((val_ground_truth_result, val_y_source), dim=0)
                    # print('prediction_result shape is', prediction_result.shape)
                    # print('ground_truth_result shape is', ground_truth_result.shape)
                    val_avg_loss = val_total_loss / (step_val + 1)
                    now_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    print("val avg loss is", val_avg_loss, "lr is", now_lr)
                    val_results = cal_metrics(val_prediction_result.cpu().numpy(), val_ground_truth_result.cpu().numpy())
                    val_loss_list[epoch] = val_avg_loss
                    # print('validation result')
                    # format_print(val_results)
                    scheduler.step(val_avg_loss)
                # Testing ------------------------------------------------------------
                if epoch  == args.epochs_pre-1:
                    model.eval()
                    with torch.no_grad():
                        for step_test, (data_source, data_target) in enumerate(test_dataloader_combined):
                            # print('test step is', step_test)
                            data_source = data_source.to(device)
                            data_target = data_target.to(device)
                            
                            test_y_source = data_source.y.squeeze(-1)
                            test_y_traget = data_target.y.squeeze(-1)
                            # forward
                            output_predict_source, output_predict_target, featrue_output_source, featrue_output_target, source_both_featrue, target_both_featrue= model(data_source, data_target, args.alpha)
                            # loss calculate
                            loss_predict_source = criterion(output_predict_source, test_y_source)
                            loss_predict_traget = criterion(output_predict_target, test_y_traget)
                            loss_featrue_source = criterion(featrue_output_source, source_both_featrue)
                            loss_featrue_traget = criterion(featrue_output_target, target_both_featrue)
                            loss_featrue_both = criterion(target_both_featrue, source_both_featrue)
                            test_loss = loss_predict_source + loss_predict_traget 
                            test_total_loss = test_total_loss + test_loss.item()
                            if step_test == 0:
                                test_prediction_result = output_predict_source
                                test_ground_truth_result = test_y_source
                            else:
                                test_prediction_result = torch.cat((test_prediction_result, output_predict_source), dim=0)
                                test_ground_truth_result = torch.cat((test_ground_truth_result, test_y_source), dim=0)
                            
                            
                    # print('prediction_result shape is', prediction_result.shape)
                    # print('ground_truth_result shape is', ground_truth_result.shape)
                    test_avg_loss = test_total_loss / (step_test + 1)
                    print("test test loss is", test_avg_loss)
                    test_results = cal_metrics(test_prediction_result.cpu().numpy(), test_ground_truth_result.cpu().numpy())
                    test_loss_list[epoch] = test_avg_loss
                    # print('test result')
                    # format_print(test_results)

                    if epoch == 0:
                        best_results = test_results
                        best_results_epoch = epoch
                    else:
                        if np.sum(test_results['MAE']) < np.sum(best_results['MAE']):
                            best_results = test_results
                            best_results_epoch = epoch
                        # print("========== New record result1 ==========")
                    if epoch == args.epochs_pre-1:
                        print("========== Best record result ==========")
                        print("best_results_epoch for pre-train is ", best_results_epoch,'alpha is', args.alpha)
                        format_print(best_results)
                        tmp = pd.DataFrame(best_results)
                        tmp = tmp.T
                        tmp.to_csv('/home/zzy6anl/cgy/mytry/result/mymodel/alpha{}.csv'.format(args.alpha))
                        
            else:
                # print('now finetuned',epoch,'epochs')
                len_dataloader = len(train_dataloader_combined)
                for step, data in enumerate(train_dataloader_combined):
                    # print('step',step,'data',data)
                    train_y_source = data.y.to(device).squeeze(-1)
                    train_y_label = data.label.to(device)
                    train_y_label = torch.reshape(train_y_label, (-1, args.label_num))
                    data = data.to(device)
                    p = float(step + epoch * len_dataloader) / args.epochs / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    train_prediction_data, train_classify_label= model(data, alpha)
                    #print('train_prediction_data shape is ',train_prediction_data.shape)
                    optimizer.zero_grad()
                    loss = criterion(train_prediction_data, train_y_source) + criterion(train_classify_label, train_y_label)
                    loss.backward()
                    optimizer.step()
                    total_loss = total_loss + loss.item()

                avg_loss = total_loss / (step + 1)
                end_time = time.time()
                # print('prediction_data shape is', prediction_data.shape)
                # print('output_data shape is', output_data.shape)
                print("Epoch:", epoch, "/", args.epochs, "Training time is ", end_time - start_time, "avg loss is",
                    avg_loss)
                train_loss_list[epoch] = avg_loss

                # validation
                if epoch % 1 == 0:
                    model.eval()
                    with torch.no_grad():
                        for step_val, data in enumerate(val_dataloader_combined):
                            # print('test step is', step_test)
                            val_output_data = data.y.to(device).squeeze(-1)
                            data = data.to(device)
                            val_y_label = data.label.to(device)
                            val_y_label = torch.reshape(val_y_label, (-1, args.label_num))
                            p = float(step + epoch * len_dataloader) / args.epochs / len_dataloader
                            alpha = 2. / (1. + np.exp(-10 * p)) - 1
                            val_prediction_data, val_classify_label = model(data, alpha)
                            val_loss = criterion(val_prediction_data, val_output_data)+ criterion(val_classify_label, val_y_label)
                            val_total_loss = val_total_loss + val_loss.item()
                            val_prediction_data = torch.reshape(val_prediction_data, (-1,args.prediction_len))
                            val_output_data = torch.reshape(val_output_data, (-1,args.prediction_len))
                            if step_val == 0:
                                val_prediction_result_combined = val_prediction_data
                                val_ground_truth_result_combined = val_output_data
                            else:
                                val_prediction_result_combined = torch.cat((val_prediction_result_combined, val_prediction_data), dim=0)
                                val_ground_truth_result_combined = torch.cat((val_ground_truth_result_combined, val_output_data), dim=0)
                    #print('prediction_result shape is', prediction_result.shape)
                    #print('ground_truth_result shape is', ground_truth_result.shape)
                    val_avg_loss = val_total_loss / (step_val + 1)
                    now_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    print("val avg loss is", val_avg_loss, "lr is", now_lr)
                    val_results = cal_metrics(val_prediction_result_combined.cpu().numpy(), val_ground_truth_result_combined.cpu().numpy())
                    val_loss_list[epoch] = val_avg_loss
                    # print('validation result')
                    # format_print(val_results)
                    scheduler.step(val_avg_loss)
                # Testing ------------------------------------------------------------
                if epoch % 1 == 0:
                    model.eval()
                    with torch.no_grad():
                        for step_test, data in enumerate(test_dataloader_combined):
                            # print('test step is', step_test)
                            test_output_data = data.y.to(device).squeeze(-1)
                            data = data.to(device)
                            test_y_label = data.label.to(device)
                            test_y_label = torch.reshape(test_y_label, (-1, args.label_num))
                            # print('input_data shape is', input_data.shape)
                            # test_input_data = torch.reshape(test_input_data, (-1, args.history_len, station_number_source * feature_dim))
                            # print('input_data shape is', input_data.shape)
                            # test_output_data = torch.reshape(test_output_data, (-1, args.prediction_len, station_number_source))
                            p = float(step + epoch * len_dataloader) / args.epochs / len_dataloader
                            alpha = 2. / (1. + np.exp(-10 * p)) - 1
                            test_prediction_data, test_classify_label = model(data, alpha)
                            test_loss = criterion(test_prediction_data, test_output_data)+ criterion(test_classify_label, test_y_label)
                            test_total_loss = test_total_loss + test_loss.item()
                            test_prediction_data = torch.reshape(test_prediction_data, (-1,args.prediction_len))
                            test_output_data = torch.reshape(test_output_data, (-1,args.prediction_len))
                            if step_test == 0:
                                test_prediction_result_combined = test_prediction_data
                                test_ground_truth_result_combined = test_output_data
                            else:
                                test_prediction_result_combined = torch.cat((test_prediction_result_combined, test_prediction_data), dim=0)
                                test_ground_truth_result_combined = torch.cat((test_ground_truth_result_combined, test_output_data), dim=0)
                    # print('prediction_result shape is', prediction_result.shape)
                    # print('ground_truth_result shape is', ground_truth_result.shape)
                    test_avg_loss = test_total_loss / (step_test + 1)
                    print("test avg loss is", test_avg_loss)
                    test_results_combined = cal_metrics(test_prediction_result_combined.cpu().numpy(), test_ground_truth_result_combined.cpu().numpy())
                    test_loss_list[epoch] = test_avg_loss
                    # print('test result')
                    # format_print(test_results)

                    if epoch-args.epochs_pre == 0:
                        best_results_combined = test_results_combined
                        best_results_epoch_combined = epoch
                    else:
                        if np.sum(test_results_combined['MAE']) < np.sum(best_results_combined['MAE']):
                            best_results_combined = test_results_combined
                            best_results_epoch_combined = epoch
                        # print("========== New record result1 ==========")
            print('-----------------------------------------------------')
        np.savetxt('/home/zzy6anl/cgy/mytry/train_loss.csv', train_loss_list)
        np.savetxt('/home/zzy6anl/cgy/mytry/val_loss.csv', val_loss_list)
        np.savetxt('/home/zzy6anl/cgy/mytry/test_loss.csv', test_loss_list)
        print('loss save done')
        writer.close()
        # print("========== Best record result ==========")
        # print("best_results_epoch for fine-tune is ", best_results_epoch_combined)
        # format_print(best_results_combined)
        


    except KeyboardInterrupt:
        writer.close()
        print("========== Best record result ==========")
        format_print(best_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_len', default=24, type=int,
                        help='How many time series sequence are used for prediction.')
    parser.add_argument('--prediction_len', default=6, type=int, help='How many time series sequence to predict')
    parser.add_argument('--gat_in_channel', default=16, type=int, help='gat input channel')
    parser.add_argument('--gat_out_channel', default=32, type=int, help='gat output channel')
    parser.add_argument('--gru_input_size', default=1, type=int, help='gat input channel')
    parser.add_argument('--gru_hidden_size', default=16, type=int, help='gru hidden_size')
    parser.add_argument('--output_dim', default=1, type=int, help='Predictor output_dim')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--patience', default=15, type=int, help='lr_scheduler patience')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--epochs_pre', default=30, type=int)
    parser.add_argument('--label_num', default=2, type=int)
    parser.add_argument('--alpha', default=50, type=int)
    parser.add_argument('--loss_alpha', default=-10, type=int)
    args = parser.parse_args()
    print("========== arguments ==========")
    print(args)
    print("================================")
    main(args)
