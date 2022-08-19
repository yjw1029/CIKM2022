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
            hidden=64,
            max_depth=2,
            dropout=0.2,
            gnn=self.args.model_cls,
            pooling="mean",
        )

        self.model = self.model.cuda()

    def init_aggregator(self):
        self.agg = get_agg_cls(self.args.agg_cls)(self.args, self.model)

    def init_client_sampler(self):
        if self.args.clients_per_step < self.args.clients_num:
            g = random.Random(self.args.client_sample_seed)
            return functools.partial(
                g.sample,
                list(range(1, self.args.clients_num + 1)),
                self.args.clients_per_step,
            )
        elif self.args.clients_per_step == self.args.clients_num:
            return lambda x: x
        else:
            raise ValueError(
                f"clients_per_step should be smaller clients_num. get {self.args.clients_per_step} and {self.args.clients_num}."
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
