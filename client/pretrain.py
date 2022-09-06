from client.flreco import FLRecoClient,FLRecoRGNNClient
import numpy as np
from collections import defaultdict
from model import get_model_cls
import torch.nn as nn
import torch
from metrics import get_metric
from .rgnn import HashTensorWrapper
import logging


def feature_extractor(layer_data, graph):
    indxs   = {}
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        indxs[_type]   = idxs
    return indxs

class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        self.times[time] = True
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return ['mol']



def sample_subgraph(graph, time_range, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor =feature_extractor):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id: [ser, time]}
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))
    new_layer_adj  = defaultdict( #target_type
                                    lambda: defaultdict(  #source_type
                                        lambda: defaultdict(  #relation_type
                                            lambda: [] #[target_id, source_id]
                                )))
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if source_time > np.max(list(time_range.keys())) or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time

    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
    '''
    indxs= feature_extractor(layer_data, graph)
            
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld  = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    return edge_list, indxs

def to_torch(feature, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    edge_index   = []
    edge_type    = []
    
    node_num = 0
    types = graph.get_types()

    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num     += len(feature[t])
        
    for t in types:
        node_feature += list(feature[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [relation_type]   
    node_feature = torch.FloatTensor(np.array(node_feature))
    node_type    = torch.LongTensor(node_type)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_index, edge_type, node_dict




class PretrainClient(FLRecoRGNNClient):
    '''Refer to https://arxiv.org/abs/2006.15437.
     GPT-GNN: Generative Pre-Training of Graph Neural Networks.
    '''
    def __init__(self, args, client_config, uid):
        super().__init__(args, client_config, uid)


    def gpt_sample(self,data):
        batch_size = data.data_index.shape[0]
        samp_target_nodes = []
        graph = Graph()
        for edge_idx,(i,j) in enumerate(data.edge_index.t()):
            node_i={'id':i.item(),'type':'mol','attr':data.x[i]}
            node_j={'id':j.item(),'type':'mol','attr':data.x[j]}
            graph.add_edge(node_i,node_j,relation_type=data.edge_type[edge_idx].item())

        for i in range(batch_size):
            node_list = list(range(data.ptr[i],data.ptr[i+1]))
            target_node = np.random.choice(node_list, self.args.sample_node_num)
            samp_target_nodes.append([target_node[0],1])

        threshold = 0.5
        edge_list, ori_idx = sample_subgraph(graph, {1: True}, \
                inp = {'mol': samp_target_nodes},  \
                    sampled_depth = self.args.sample_depth, sampled_number = self.args.sample_neighbor_number)
            
        rem_edge_list = defaultdict(  #source_type
                        lambda: defaultdict(  #relation_type
                            lambda: [] # [target_id, source_id] 
                                ))
        ori_list = {}
        target_type='mol'
        rel_stop_list = ['self']

        for source_type in edge_list[target_type]:
            ori_list[source_type] = {}
            for relation_type in edge_list[target_type][source_type]:
                ori_list[source_type][relation_type] = np.array(edge_list[target_type][source_type][relation_type])
                el = []
                for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                    if target_ser < source_ser:
                        if relation_type not in rel_stop_list and target_ser < batch_size and np.random.random() > threshold:
                            rem_edge_list[source_type][relation_type] += [[target_ser, source_ser]]
                            continue
                        el += [[target_ser, source_ser]]
                        el += [[source_ser, target_ser]]
                el = np.array(el)
                edge_list[target_type][source_type][relation_type] = el
                    
        '''
            Adding feature nodes:
        '''
        feature = {}
        node_cls = {}
        feature['mol'] = data.x[ori_idx['mol']]
        node_cls['mol'] = data.node_type[ori_idx['mol']]
        n_target_nodes = len(feature[target_type])
        feature[target_type] = np.concatenate((feature[target_type], np.zeros([batch_size, feature[target_type].shape[1]])))

        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                el = []
                for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                    if target_ser < batch_size:
                            el += [[target_ser + n_target_nodes,source_ser]]
                if len(el) > 0:
                    edge_list[target_type][source_type][relation_type] = \
                        np.concatenate((edge_list[target_type][source_type][relation_type], el))


        rem_edge_lists = {}

        for source_type in rem_edge_list:
            rem_edge_lists[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                rem_edge_lists[source_type][relation_type] = np.array(rem_edge_list[source_type][relation_type])
        del rem_edge_list
            
        return to_torch(feature, edge_list, graph), rem_edge_lists, ori_list, \
                (n_target_nodes, n_target_nodes + batch_size),node_cls

    def train(self,reset_optim=True):
        if reset_optim:
            self.reset_optimizer()

        self.model = self.model.cuda()
        self.model.train()

        #self.reconstruct()

        local_training_steps = 0
        local_training_num = 0

        target_type='mol'
        loss_cum = 0
        for epoch in range(self.args.local_epoch):
            for data in self.dataloader_dict["train"]:
                self.optimizer.zero_grad()
                data_sample, rem_edge_list, ori_edge_list, (start_idx, end_idx), node_cls = self.gpt_sample(data)
                node_feature, node_type, edge_index, edge_type, node_dict = data_sample
                data.x = node_feature
                data.edge_index=edge_index
                data.edge_type=edge_type
                data = data.cuda()
                node_emb = self.model(data,start_idx,end_idx)
                loss_link = self.model.link_loss(node_emb, rem_edge_list, ori_edge_list, node_dict, target_type)
                loss_attr = self.model.feat_loss(node_emb[start_idx : end_idx], node_cls['mol'][:data.data_index.shape[0]].cuda())
                loss =  loss_attr * self.args.attr_ratio + loss_link[0] * (1 - self.args.attr_ratio)
                loss_cum += loss
                if loss != 0:
                    loss.backward()
                self.optimizer.step()

                local_training_steps += 1
                local_training_num += data.data_index.shape[0] + loss_link[2]
        print(loss_cum)
        return local_training_num, local_training_steps
    
    @torch.no_grad()
    def eval(self, dataset="val"):
        self.model = self.model.cuda()
        self.model.eval()

        for data in self.dataloader_dict[dataset]:
            data_sample, rem_edge_list, ori_edge_list, (start_idx, end_idx), node_cls = self.gpt_sample(data)
            node_feature, node_type, edge_index, edge_type, node_dict= data_sample
            data.x = node_feature
            data.edge_index=edge_index
            data.edge_type=edge_type
            data = data.cuda()
            target_type='mol'
            node_emb = self.model(data,start_idx,end_idx)
            loss_link = self.model.link_loss(node_emb, rem_edge_list, ori_edge_list, node_dict, target_type)
            loss_attr = self.model.feat_loss(node_emb[start_idx : end_idx], node_cls['mol'][:data.data_index.shape[0]].cuda())
            loss = loss_link[0] * (1 - self.args.attr_ratio) + loss_attr * self.args.attr_ratio

            for metric in self.metric_cals:
                if metric == 'sum_loss':
                    self.metric_cals[metric].update(loss.item())
                elif metric == "node_loss":
                    self.metric_cals[metric].update(loss_attr.item())
                elif metric == "edge_loss":
                    self.metric_cals[metric].update(loss_link[0].item() if loss_link[0]!=0 else 0)

        rslt = {}
        for metric in self.metric_cals:
            if metric != "relative_impr":
                rslt[metric] = self.metric_cals[metric].compute()
        
        # clear state in metrics
        for metric in self.metric_cals:
            if metric != "relative_impr":
                self.metric_cals[metric].clear()

        return rslt

    @torch.no_grad()
    def eval_finetune(self, dataset="val"):
        self.model = self.model.cuda()
        self.model.eval()

        for data in self.dataloader_dict[dataset]:
            data = data.cuda()
            pred = self.model(data)
            if "classification" in self.task_type.lower():
                label = data.y.squeeze(-1).long()
            elif "regression" in self.task_type.lower():
                label = data.y

            if len(label.size()) == 0:
                label = label.unsqueeze(0)

            for metric in self.metric_cals:
                if metric != "relative_impr":
                    self.metric_cals[metric].update(pred, label)

        rslt = {}
        for metric in self.metric_cals:
            if metric != "relative_impr":
                rslt[metric] = self.metric_cals[metric].compute()
        
        # clear state in metrics
        for metric in self.metric_cals:
            if metric != "relative_impr":
                self.metric_cals[metric].clear()

        if "relative_impr" in self.metric_cals:
            rslt["relative_impr"] = (
                self.base_metric - rslt[self.major_metric]
            ) / self.base_metric

        return rslt

    def preprocess_data(self, data):
        # add virtual node if pooling as virtual node
        data = super().preprocess_data(data)

        node_set = set()
        for split in ["train", "val", "test"]:
            for i in data[split]:
                node_set = node_set | set([HashTensorWrapper(j) for j in list(i.x)])

        node_set = list(set(node_set))
        self.node_types = len(node_set)

        logging.info(f"Client {self.uid} has {self.node_types} node type.")

        node_type_dict = {i: cnt for cnt, i in enumerate(node_set)}
        for split in ["train", "val", "test"]:
            for i in data[split]:
                i.node_type = torch.LongTensor(
                    [node_type_dict[HashTensorWrapper(j)] for j in i.x]
                )

        return data

    def init_model(self):
        model_cls = get_model_cls(self.model_cls)
        self.model = model_cls(
            self.in_channels,
            self.out_channels,
            num_relations=self.num_relations,
            hidden=self.hidden,
            max_depth=self.max_depth,
            dropout=self.dropout,
            pooling=self.pooling,
            num_bases=self.num_bases,
            base_agg=self.base_agg,
            root_weight=True,
            mode='pretrain',
            node_types=self.node_types
        )

        if "classification" in self.task_type.lower():
            self.criterion = nn.CrossEntropyLoss()
        elif "regression" in self.task_type.lower():
            self.criterion = nn.MSELoss()
