import math
import random
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class GATGRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gat_in_channel = args.gat_in_channel
        self.gat_out_channel = args.gat_out_channel
        self.hidden_dim = args.gru_hidden_size
        self.output_dim = args.output_dim
        self.history_len = args.history_len
        self.predictor_len = args.prediction_len
        self.build()

    def build(self):
        self.gat_conv = GATConv(self.gat_in_channel * self.history_len,
                                self.gat_out_channel * self.history_len, heads=3, concat=False)
        self.gru_layer = nn.GRU(self.gat_out_channel, self.hidden_dim, batch_first=True)
        self.predictor_1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.predictor_2 = nn.Linear(self.history_len, self.predictor_len)

    def forward(self, data):
        # print('data.x shape is', data.x.shape)
        batch_size, node_num, seq_len, input_dim = data.x.shape
        gat_inputs = torch.reshape(data.x, (-1, self.history_len, self.gat_in_channel))
        # print('gat_inputs shape is', gat_inputs.shape)
        gat_inputs = torch.reshape(gat_inputs, (-1, self.gat_in_channel * self.history_len))
        # print('[]gat_inputs shape is', gat_inputs.shape)
        gat_outputs = self.gat_conv(gat_inputs, data.edge_index)
        # print('gat_outputs shape is', gat_outputs.shape)
        gru_input = torch.reshape(gat_outputs, (-1, self.history_len, self.gat_out_channel))
        # print('gru_inputs shape is', gru_input.shape)
        gru_outputs, _ = self.gru_layer(gru_input)
        # print('gru_outputs shape is', gru_outputs.shape)
        output = self.predictor_1(gru_outputs).squeeze(-1)
        # print('output shape is', output.shape)
        output = torch.reshape(self.predictor_2(output), (batch_size, node_num, -1))
        # print('output shape is', output.shape)
        return output.permute(0, 2, 1)

    """
    Step1:构建Dataset，包括data.x, data.y, data.edge_index, data.edge_attr
    data.x shape is [batch_size, node_num, seq_len, input_dim]
    data.edge_index shape is [2, |E|]
    data.edge_attr shape is [|E|, edge_feature_dim]
    Step2:Dataloader封装数据
    Step3 to StepN 训练验证测试
    """
class GRUGAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gru_input_size = args.gru_input_size
        # self.gat_in_channel = args.gat_in_channel
        self.gru_hidden_size = args.gru_hidden_size
        self.gat_out_channel = args.gat_out_channel
        self.output_dim = args.output_dim
        self.history_len = args.history_len
        self.predictor_len = args.prediction_len
        self.batch_size = args.batch_size
        self.build()

    def build(self):
        self.gru_layer = nn.GRU(self.gru_input_size, self.gru_hidden_size, batch_first=True)
        self.gat_conv = GATConv(self.gru_hidden_size * self.history_len,
                                self.gat_out_channel * self.history_len, heads=3, concat=False, edge_dim=1, add_self_loops=False)
        # self.predictor_1 = nn.Linear(self.gat_out_channel, self.gru_hidden_size)
        self.predictor_1 = nn.Linear(self.gat_out_channel, 1)
        self.predictor_2 = nn.Linear(self.gru_hidden_size, self.gru_input_size)
        self.predictor_3 = nn.Linear(self.history_len, self.predictor_len)
        self.classifer_1 = nn.Linear(self.history_len * self.gat_out_channel, self.gat_out_channel)
        self.classifer_2 = nn.Linear(self.gat_out_channel, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.GRL = ReverseLayerF()
        # self.pooling = global_max_pool()

    def forward(self, data, alpha):
        # print('data.x shape is', data.x.shape)
        batch_size, node_num, seq_len, input_dim = data.x.shape
        gru_input = torch.reshape(data.x, (-1, self.history_len, self.gru_input_size))
        #print('gru_inputs shape is', gru_input.shape)
        gru_output, _ = self.gru_layer(gru_input)
        # print('gru_output shape is', gru_output.shape)
        #gat_input = torch.reshape(gru_output, (-1, node_num, self.history_len, self.gru_hidden_size))
        # 32, node_num, 24, 16
        #gat_input = torch.reshape(gat_input, (-1, self.history_len, self.gru_hidden_size))
        gat_input = torch.reshape(gru_output, (-1, self.gru_hidden_size * self.history_len))
        # print('gru_inputs shape is', gru_input.shape)
        gat_output = self.gat_conv(gat_input, data.edge_index, data.edge_attr)
        # predictor --------------------------------------------------------------------------------------------
        
        #print('gat_outputs shape is', gat_output.shape)
        linear_input = torch.reshape(gat_output, (-1, self.history_len, self.gat_out_channel))
        # print('linear inputs shape is ', linear_input.shape)
        output_predict = self.relu(self.predictor_1(linear_input)).squeeze(-1)
        output_predict = torch.reshape(self.relu(self.predictor_3(output_predict)), (batch_size, node_num, -1))
        # domain classify----------------------------------------------------------------------------------------
        classifer_input = ReverseLayerF.apply(gat_output, alpha)
        pooling_output = global_max_pool(classifer_input, data.batch)
        #print('classifer_input shape is ', classifer_input.shape)
        #print('pooling_output shape is ', pooling_output.shape)
        linear_output_1 = self.classifer_1(pooling_output)
        #print('linear_output_1 shape is ', linear_output_1.shape)
        linear_output_2 = self.classifer_2(linear_output_1)
        #print('linear_output_2 shape is ', linear_output_2.shape)
        output_label = self.softmax(linear_output_2)
        #print('output_label shape is ', output_label.shape)
        #return output.permute(0, 2, 1)
        return output_predict, output_label
# gru-gat
class mytry(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gru_input_size = args.gru_input_size
        # self.gat_in_channel = args.gat_in_channel
        self.gru_hidden_size = args.gru_hidden_size
        self.gat_out_channel = args.gat_out_channel
        self.output_dim = args.output_dim
        self.history_len = args.history_len
        self.predictor_len = args.prediction_len
        self.batch_size = args.batch_size
        self.build()
        # self.is_pretrain = args.is_pretrain

    def build(self):
        self.gru_layer_source = nn.GRU(self.gru_input_size, self.gru_hidden_size, batch_first=True)
        self.gru_layer_target = nn.GRU(self.gru_input_size, self.gru_hidden_size, batch_first=True)
        self.gru_layer_both = nn.GRU(self.gru_input_size, self.gru_hidden_size, batch_first=True)
        self.gat_conv_source = GATConv(self.gru_hidden_size * self.history_len,
                                self.gat_out_channel * self.history_len, heads=3, concat=False, edge_dim=1, add_self_loops=False)
        self.gat_conv_target = GATConv(self.gru_hidden_size * self.history_len,
                                self.gat_out_channel * self.history_len, heads=3, concat=False, edge_dim=1, add_self_loops=False)  
        self.gat_conv_both = GATConv(self.gru_hidden_size * self.history_len,
                                self.gat_out_channel * self.history_len, heads=3, concat=False, edge_dim=1, add_self_loops=False)                                              
        # self.predictor_1 = nn.Linear(self.gat_out_channel, self.gru_hidden_size)
        self.predictor_1 = nn.Linear(self.gat_out_channel, 1)
        self.predictor_2 = nn.Linear(self.gru_hidden_size, self.gru_input_size)
        self.predictor_3 = nn.Linear(self.history_len, self.predictor_len)
        self.predictor_4 = nn.Linear(self.batch_size * 2, self.batch_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.GRL = ReverseLayerF()
        
    
    
    def forward(self, data_source, data_target, alpha):
        batch_size, node_num_source, seq_len, input_dim = data_source.x.shape
        batch_size, node_num_target, seq_len, input_dim = data_target.x.shape
        # source feature extarct -----------------------------------------------------------------
        gru_input_source = torch.reshape(data_source.x, (-1, self.history_len, self.gru_input_size))
        gru_output_source, _ = self.gru_layer_source(gru_input_source)
        gat_input_source = torch.reshape(gru_output_source, (-1, self.gru_hidden_size * self.history_len))
        gat_output_source = self.gat_conv_source(gat_input_source, data_source.edge_index, data_source.edge_attr)
        # print('gat_output_source shape is ', gat_output_source.shape)
        # print('data_source.batch', data_source.batch)
        featrue_output_source = global_max_pool(gat_output_source, data_source.batch)
        featrue_output_source = torch.nn.functional.sigmoid(featrue_output_source)
        # target feature extract -----------------------------------------------------------------
        gru_input_target = torch.reshape(data_target.x, (-1, self.history_len, self.gru_input_size))
        gru_output_target, _ = self.gru_layer_target(gru_input_target)
        gat_input_target = torch.reshape(gru_output_target, (-1, self.gru_hidden_size * self.history_len))
        gat_output_target = self.gat_conv_target(gat_input_target, data_target.edge_index, data_target.edge_attr)
        featrue_output_target = global_max_pool(gat_output_target, data_target.batch)
        featrue_output_target = torch.nn.functional.sigmoid(featrue_output_target)
        # both feature extract -------------------------------------------------------------------
        gru_output_source_both, _ = self.gru_layer_both(gru_input_source)
        gat_input_source_both = torch.reshape(gru_output_source_both, (-1, self.gru_hidden_size * self.history_len))
        gat_output_source_both = self.gat_conv_both(gat_input_source_both, data_source.edge_index, data_source.edge_attr)


        gru_output_target_both, _ = self.gru_layer_both(gru_input_target)
        gat_input_target_both = torch.reshape(gru_output_target_both, (-1, self.gru_hidden_size * self.history_len))
        gat_output_target_both = self.gat_conv_both(gat_input_target_both, data_target.edge_index, data_target.edge_attr)
        # predict -------------------------------------------------------------------------------
        linear_input_source = gat_output_source + alpha * gat_output_source_both
        linear_input_source = torch.reshape(linear_input_source, (-1, self.history_len, self.gat_out_channel))
        output_predict_source = self.relu(self.predictor_1(linear_input_source)).squeeze(-1)
        output_predict_source = torch.reshape(self.relu(self.predictor_3(output_predict_source)), (batch_size, node_num_source, -1))
        # print('output_predict_source shape is ',output_predict_source.shape)
        linear_input_target = gat_output_target + alpha * gat_output_target_both
        linear_input_target = torch.reshape(linear_input_target, (-1, self.history_len, self.gat_out_channel))
        output_predict_target = self.relu(self.predictor_1(linear_input_target)).squeeze(-1)
        output_predict_target = torch.reshape(self.relu(self.predictor_3(output_predict_target)), (batch_size, node_num_target, -1))
        # print('output_predict_target shape is ',output_predict_target.shape)
        # domain -----------------------------------------------------------------------------------------------------
        source_both_featrue = global_max_pool(gat_output_source_both, data_source.batch)
        source_both_featrue = torch.nn.functional.sigmoid(source_both_featrue)
        target_both_featrue = global_max_pool(gat_output_target_both, data_target.batch)
        target_both_featrue = torch.nn.functional.sigmoid(target_both_featrue)
        return output_predict_source, output_predict_target, featrue_output_source, featrue_output_target, source_both_featrue, target_both_featrue
