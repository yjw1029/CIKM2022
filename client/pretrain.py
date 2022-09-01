from client.flreco import FLRecoClient

class PretrainClient(FLRecoClient):
    '''Refer to https://arxiv.org/abs/2006.15437.
     GPT-GNN: Generative Pre-Training of Graph Neural Networks.
    '''
    def __init__(self, args, client_config, uid):
        super().__init__(args, client_config, uid)

    def preprocess_data(self, data):
        if self.pooling == "virtual_node":
            logging.info("[+] Apply virtual node. Preprocessing data")
            transform = VirtualNode()
            for split in ["train", "val", "test"]:
                data[split] = list(map(transform, data[split]))
        return data
