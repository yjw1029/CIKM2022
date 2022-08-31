import random
import functools

from agg import get_agg_cls
from model import get_model_cls


class BaseServer:
    def __init__(self, args):
        self.args = args

        self.init_model()
        self.init_aggregator()

        self.client_sampler = self.init_client_sampler()
        self.clear_rslt()

    def init_model(self):
        model_cls = get_model_cls(self.args.model_cls)

        # dummy input and output channel
        self.model = model_cls(
            1,
            1,
            hidden=self.args.hidden,
            max_depth=self.args.max_depth,
            dropout=self.args.dropout,
            pooling=self.args.pooling,
            num_bases=self.args.num_bases,
            num_relations=0,
            base_agg=self.args.base_agg,
            share_emb=self.args.share_emb,
            cluster_num=self.args.cluster_num
        )

        self.model = self.model.cuda()

    def init_aggregator(self):
        self.agg = get_agg_cls(self.args.agg_cls)(self.args, self.model)

    def init_client_sampler(self):
        if self.args.clients_per_step < len(self.args.clients):
            g = random.Random(self.args.client_sample_seed)
            return functools.partial(
                g.sample,
                self.args.clients,
                self.args.clients_per_step,
            )
        elif self.args.clients_per_step == len(self.args.clients):
            return lambda x: x
        else:
            raise ValueError(
                f"clients_per_step should be smaller clients_num. get {self.args.clients_per_step} and {len(self.args.clients)}."
            )

    def sample_clients(self):
        return self.client_sampler()

    def clear_rslt(self):
        self.clients_rslts = {}

    def collect(self, uid, client_rslt):
        self.clients_rslts[uid] = client_rslt

    def update(self, step):
        agg_rslt = self.agg.aggregate(self.clients_rslts)
        self.clear_rslt()

        return agg_rslt
