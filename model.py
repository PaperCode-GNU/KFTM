import os
import torch
import torch.nn as nn
import numpy as np

from sentence_transformers import SentenceTransformer, util
from transformers import BertModel
from data_loader import get_num_class
from topjudge_parser import ConfigParser

class Diff_LawCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert (self.hidden_size % self.num_heads == 0), "hidden size must be divisible by num heads"

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

        self.lambda_init = 0.2
        self.lambda_q1 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32).normal_(mean=0, std=0.1))

    def forward(self, fact_embed, law_embed, mask = None):
        batch_size = fact_embed.size(0)

        Q = self.query(fact_embed)   
        # print("Q size:", Q.shape)
        K = self.key(law_embed)   
        V = self.value(law_embed)  

        # 多头
        Q1 = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        K1 = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        Q2 = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        K2 = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  

        scores_1 = torch.matmul(Q1, K1.transpose(-2, -1)) / np.sqrt(self.head_dim)  
        attention_weights_1 = nn.functional.softmax(scores_1, dim=-1)
        scores_2 = torch.matmul(Q2, K2.transpose(-2, -1)) / np.sqrt(self.head_dim)  
        attention_weights_2 = nn.functional.softmax(scores_2, dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attention_weights = attention_weights_1 - lambda_full * attention_weights_2
        context = torch.matmul(attention_weights, V)   
        context = context.transpose(1, 2).contiguous().view(batch_size, 500, -1)
        # print(context.shape)
        attention_output = self.out(context)

        return attention_output

class LawCrossAttention(nn.Module):
    def __init__(self, config):
        super(LawCrossAttention, self).__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert (self.hidden_size % self.num_heads == 0), "hidden size must be divisible by num heads"

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, fact_embed, law_embed, mask = None):
        batch_size = fact_embed.size(0)
        fact_embed = fact_embed  
        law_embed = law_embed  

        Q = self.query(fact_embed) 
        K = self.key(law_embed)   
        V = self.value(law_embed)

        # 多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim) 
        attention_weights = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V) 

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attention_output = self.out(context)

        return attention_output

class TopJudge(nn.Module):
    def __init__(self):
        super(TopJudge, self).__init__()
        configFilePath = './GCN.config'
        config_lstm = ConfigParser(configFilePath)
        self.config = config_lstm

        self.num1 = 103
        self.num2 = 119

        # self.num1 = 172
        # self.num2 = 190

        #self.feature_len = config.getint("net", "hidden_size")

        features = self.config.getint("net", "hidden_size")  #hidden_size = 768
        self.hidden_dim = features  #768
        self.outfc = []
        task_name = self.config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in task_name:
            self.outfc.append(nn.Linear(features, get_num_class(x, self.num1, self.num2)
            ))  #这是最终预测的任务

        self.midfc = []#
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in task_name:
            self.cell_list.append(nn.LSTMCell(input_size = 768, hidden_size = features))

        self.hidden_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.hidden_list = []
        for a in range(0, len(task_name) + 1):
            self.hidden_list.append((
                torch.zeros(self.config.getint("data", "batch_size"), self.hidden_dim, requires_grad=True).cuda(),
                torch.zeros(self.config.getint("data", "batch_size"), self.hidden_dim, requires_grad=True).cuda()))  # [32, 512]

        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)


    def generate_graph(self):
        s = self.config.get("data", "graph")
        arr = s.replace("[", "").replace("]", "").split(",")
        graph = []
        n = 0
        if (s == "[]"):
            arr = []
            n = 3
        for a in range(0, len(arr)):
            arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
            arr[a][0] = int(arr[a][0])
            arr[a][1] = int(arr[a][1])
            n = max(n, max(arr[a][0], arr[a][1]))

        n += 1
        for a in range(0, n):
            graph.append([])
            for b in range(0, n):
                graph[a].append(False)

        for a in range(0, len(arr)):
            graph[arr[a][0]][arr[a][1]] = True

        return graph

    def forward(self, x):
        fc_input = x    #[32, 768]
        outputs = []
        task_name = self.config.get("data", "type_of_label").replace(" ", "").split(",")
        graph = self.generate_graph()

        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(task_name) + 1):
                if graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)
            # self.hidden_list[a] = h, c
            if self.config.getboolean("net", "more_fc"):
                outputs.append(
                    self.outfc[a - 1](nn.functional.relu(self.midfc[a - 1](h))).view(self.config.getint("data", "batch_size"), -1))
            else:
                outputs.append(self.outfc[a - 1](h).view(self.config.getint("data", "batch_size"), -1))

        return outputs

class Tdf(nn.Module):
    def __init__(self, config):
        super(Tdf, self).__init__()
        self.config = config
        self.task_name = ['law', 'accu', 'term']
        self.num_law = self.config.num_law
        self.num_accu = self.config.num_accu
        self.hidden_dim = self.config.hidden_size

        # 最终预测层
        self.outfc = []
        for x in self.task_name:
            self.outfc.append(
                nn.Linear(in_features = self.hidden_dim, out_features = get_num_class(x, self.num_law, self.num_accu))
            )

        #LSTMCell层
        self.cell_list = []
        for x in self.task_name:
            self.cell_list.append(
                nn.LSTMCell(input_size = self.hidden_dim, hidden_size = self.hidden_dim)
            )

        # h隐藏状态
        self.hidden_state_fc_list = []
        for a in range(0, len(self.task_name)):
            arr = []
            for b in range(0, len(self.task_name)):
                arr.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        # c隐藏状态
        self.cell_state_fc_list = []
        for a in range(0, len(self.task_name)):
            arr = []
            for b in range(0, len(self.task_name)):
                arr.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.outfc = nn.ModuleList(self.outfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def forward(self, fact law_labels, accu_labels, label):
        batch_size = fact.size(0)
        # fc_input = input

        hidden_list = []
        for b in range(0, len(self.task_name)):
            hidden_list.append(
                (
                    torch.zeros(batch_size, self.hidden_dim).cuda(),
                    torch.zeros(batch_size, self.hidden_dim).cuda()
                )
            )
        #distinguish between training and testing
        # if label == "train":
        h_law, c_law = self.cell_list[0](fact, hidden_list[0])
        output_law = self.outfc[0](h_law)
        # output_law = self.outfc[0](nn.functional.relu(h_law))

        law_probs = nn.functional.softmax(output_law, dim = 1)
        predict_law = torch.argmax(law_probs, dim = 1)
        predict_law = predict_law.cpu().numpy()
        law_label = law_labels.cpu().numpy()
        res_law = np.equal(predict_law, law_label)

        hp_accu, cp_accu = hidden_list[1]
        hp_term_1, cp_term_1 = hidden_list[2]

        mask_1 = torch.zeros_like(hp_accu, dtype = torch.bool).cuda()
        for i in range(batch_size):
            if res_law[i] == True:
                mask_1[i, :] = True

        hp_accu = hp_accu + self.hidden_state_fc_list[0][1](h_law) * mask_1
        cp_accu = cp_accu + self.cell_state_fc_list[0][1](c_law) * mask_1
        hp_term_1 = hp_term_1 + self.hidden_state_fc_list[0][2](h_law) * mask_1
        cp_term_1 = cp_term_1 + self.cell_state_fc_list[0][2](c_law) * mask_1

        hidden_list[1] = (hp_accu, cp_accu)
        hidden_list[2] = (hp_term_1, cp_term_1)

        h_accu, c_accu = self.cell_list[1](fact, hidden_list[1])
        output_accu = self.outfc[1](h_accu)
        # output_accu = self.outfc[1](nn.functional.relu(h_accu))

        accu_probs = nn.functional.softmax(output_accu, dim = 1)
        predict_accu = torch.argmax(accu_probs, dim = 1)
        predict_accu = predict_accu.cpu().numpy()
        accu_label = accu_labels.cpu().numpy()
        res_accu = np.equal(predict_accu, accu_label)

        hp_term_2, cp_term_2 = hidden_list[2]

        mask_2 = torch.zeros_like(hp_term_2, dtype = torch.bool).cuda()
        for j in range(batch_size):
            if res_accu[j] == True:
                mask_2[j, :] = True
                
        hp_term_2 = hp_term_2 + self.hidden_state_fc_list[1][2](h_accu) * mask_2
        cp_term_2 = cp_term_2 + self.cell_state_fc_list[1][2](c_accu) * mask_2

        hidden_list[2] = (hp_term_2, cp_term_2)

        h_term, c_term = self.cell_list[2](fact, hidden_list[2])
        output_term = self.outfc[2](h_term)
        # output_term = self.outfc[2](nn.functional.relu(h_term))
        
        # else:
        #     h_law, c_law = self.cell_list[0](fact, hidden_list[0])
        #     output_law = self.outfc[0](h_law)

        #     hp_accu, cp_accu = hidden_list[1]
        #     hp_term_1, cp_term_1 = hidden_list[2]

        #     hp_accu = hp_accu + self.hidden_state_fc_list[0][1](h_law)
        #     cp_accu = cp_accu + self.cell_state_fc_list[0][1](c_law)
        #     hp_term_1 = hp_term_1 + self.hidden_state_fc_list[0][2](h_law)
        #     cp_term_1 = cp_term_1 + self.cell_state_fc_list[0][2](c_law)

        #     hidden_list[1] = (hp_accu, cp_accu)
        #     hidden_list[2] = (hp_term_1, cp_term_1)

        #     h_accu, c_accu = self.cell_list[1](fact, hidden_list[1])
        #     output_accu = self.outfc[1](h_accu)

        #     hp_term_2, cp_term_2 = hidden_list[2]

        #     hp_term_2 = hp_term_2 + self.hidden_state_fc_list[1][2](h_accu)
        #     cp_term_2 = cp_term_2 + self.cell_state_fc_list[1][2](c_accu)

        #     hidden_list[2] = (hp_term_2, cp_term_2)

        #     h_term, c_term = self.cell_list[2](fact, hidden_list[2])
        #     output_term = self.outfc[2](h_term)

        return output_law, output_accu, output_term

class LKDF(nn.Module):
    def __init__(self, config):
        super(LKDF, self).__init__()
        self.config = config
        self.act_func = nn.ReLU()
        self.lert = BertModel.from_pretrained("hfl/chinese-lert-base")
        # self.lawcrossattention = LawCrossAttention(self.config)
        self.lawcrossattention = Diff_LawCrossAttention(self.config)
        self.tdf = Tdf(self.config)
        self.topjudge = TopJudge()

        self.conv_law = nn.Conv1d(
            in_channels=768,      
            out_channels=256,     
            kernel_size=3,           
            stride=1,
            padding=1,
            dilation=1
        )
        self.conv_charge = nn.Conv1d(
            in_channels=768,      
            out_channels=256,     
            kernel_size=3,           
            stride=1,
            padding=1,
            dilation=1
        )
        
        # self.decoder_law = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.num_law)
        self.law_linear = nn.Linear(in_features = int((256 * 103) / self.config.batch_size), out_features = self.config.hidden_size)    #需要修改分类类别
        self.charge_linear = nn.Linear(in_features = int((256 * 119) / self.config.batch_size), out_features = self.config.hidden_size)#需要修改分类类别
        self.law_charge_linear = nn.Linear(in_features = self.config.hidden_size * 2, out_features = self.config.hidden_size)

        self.attention_linear = nn.Linear(in_features=self.config.hidden_size * 500, out_features=self.config.hidden_size)#需要修改长度

        self.decoder_1 = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.num_law)
        self.decoder_2 = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.num_accu)
        self.decoder_3 = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.num_term)

        #定义门控选择层
        self.gat_fc_law = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.hidden_size)
        self.gat_fc_accu = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.hidden_size)
        self.gat_fc_term = nn.Linear(in_features = self.config.hidden_size, out_features = self.config.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, input_ids, attention_mask, token_type_ids = None, position_ids = None, input_embeds = None, head_mask = None):
        outputs = self.lert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = input_embeds
        )
        return outputs

    def forward(self, fact_input_ids, fact_attention_mask, law_input_ids, law_attention_mask, charge_input_ids, charge_attention_mask, law_labels, accu_labels, train_eval_label = "train"):
        batch_size = fact_input_ids.size(0)
        fact_embed = self.encoder(fact_input_ids, fact_attention_mask).last_hidden_state
        resudial = fact_embed

        law_embed = self.encoder(law_input_ids, law_attention_mask).last_hidden_state
        law_embed = law_embed.permute(0,2,1)
        law_embed_cnn  = self.conv_law(law_embed)
        law_embed = law_embed_cnn.view(batch_size, 256, -1)
        law_embed = self.act_func(self.law_linear(law_embed)) 

        charge_embed = self.encoder(charge_input_ids, charge_attention_mask).last_hidden_state
        charge_embed = charge_embed.permute(0,2,1)
        charge_embed_cnn = self.conv_charge(charge_embed)
        charge_embed = charge_embed_cnn.view(batch_size, 256, -1)
        charge_embed = self.act_func(self.charge_linear(charge_embed))

        law_accu_embed = torch.cat([law_embed, charge_embed], dim=2)
        law_accu_embed = self.act_func(self.law_charge_linear(law_accu_embed))

        attention_output = self.lawcrossattention(fact_embed, law_accu_embed)  

        fact_attention_output = attention_output + resudial #[bh, len, hid]
        fact_attention_output = fact_attention_output.view(batch_size, -1)
        fact_final = self.act_func(self.attention_linear(fact_attention_output))


        # #门控选择
        # fact_law = self.sigmoid(self.gat_fc_law(fact_attention_output))
        # fact_accu = self.sigmoid(self.gat_fc_accu(fact_attention_output))
        # fact_term = self.sigmoid(self.gat_fc_term(fact_attention_output))

        output_law, output_accu, output_term = self.tdf(fact_final, law_labels, accu_labels, train_eval_label)

        # output_law = self.decoder_1(fact_attention_output)
        # output_accu = self.decoder_2(fact_attention_output)
        # output_term = self.decoder_3(fact_attention_output)

        # output_law, output_accu, output_term = self.topjudge(fact_attention_output)

        return output_law, output_accu, output_term#, mid_law