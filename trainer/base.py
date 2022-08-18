import yaml
import logging

from client import get_client_cls
from server import get_server_cls

class BaseTrainer:
    def init_clients(self):
        with open(self.args.client_config_file, 'r', encoding="utf-8") as f:
            client_configs = yaml.safe_load(f)

        client_cls = get_client_cls(self.args.client_cls)
        self.clients = {}
        for uid in range(1, self.args.clients_num + 1):
            client_config = client_configs[f"client_{uid}"]
            self.clients[uid] = client_cls(self.args, client_config, uid)
            logging.info(f"[-] finish init client_{uid}")
            logging.info(client_config)

    def init_server(self):
        server_cls = get_server_cls(self.args.server_cls)
        self.server = server_cls(self.args)