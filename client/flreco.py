from client.base import BaseClient
from client.rgnn import RGNNClient

from utils import is_name_in_list

class FLRecoClient(BaseClient):
    '''Refer to https://arxiv.org/abs/2102.03448.
     Federated Reconstruction: Partially Local Federated Learning.
    '''
    def freeze_shared_params(self):
        # freeze global shared parameters
        for name, param in self.model.named_parameters():
            if not is_name_in_list(name, self.args.param_filter_list):
                param.requires_grad = False

    def unfreeze_all_params(self):
        # unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

    def reconstruct(self):
        self.freeze_shared_params()

        step_cnt = 0
        while step_cnt < self.args.reco_steps:
            for data in self.dataloader_dict["train"]:
                if step_cnt >= self.args.reco_steps:
                    break
                self.optimizer.zero_grad()
                data = data.cuda()
                pred = self.model(data)
                if "classification" in self.task_type.lower():
                    label = data.y.squeeze(-1).long()
                elif "regression" in self.task_type.lower():
                    label = data.y

                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(pred, label)

                loss.backward()
                self.optimizer.step()
                step_cnt += 1
        
        self.unfreeze_all_params()

        
    def train(self, reset_optim=True):
        if reset_optim:
            self.reset_optimizer()

        self.model = self.model.cuda()
        self.model.train()

        self.reconstruct()

        local_training_steps = 0
        local_training_num = 0

        for epoch in range(self.args.local_epoch):
            for data in self.dataloader_dict["train"]:
                self.optimizer.zero_grad()
                data = data.cuda()
                pred = self.model(data)
                if "classification" in self.task_type.lower():
                    label = data.y.squeeze(-1).long()
                elif "regression" in self.task_type.lower():
                    label = data.y

                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(pred, label)

                loss.backward()
                self.optimizer.step()

                local_training_steps += 1
                local_training_num += label.shape[0]

        return local_training_num, local_training_steps


class FLRecoRGNNClient(RGNNClient):
    def freeze_shared_params(self):
        # freeze global shared parameters
        for name, param in self.model.named_parameters():
            if not is_name_in_list(name, self.args.param_filter_list):
                param.requires_grad = False

    def unfreeze_all_params(self):
        # unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

    def reconstruct(self):
        self.freeze_shared_params()

        step_cnt = 0
        while step_cnt < self.args.reco_steps:
            for data in self.dataloader_dict["train"]:
                self.optimizer.zero_grad()
                data = data.cuda()
                pred = self.model(data)
                if "classification" in self.task_type.lower():
                    label = data.y.squeeze(-1).long()
                elif "regression" in self.task_type.lower():
                    label = data.y

                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(pred, label)

                loss.backward()
                self.optimizer.step()
                step_cnt += 1
                if step_cnt>=self.args.reco_steps:
                    break
        
        self.unfreeze_all_params()

        
    def train(self, reset_optim=True):
        if reset_optim:
            self.reset_optimizer()

        self.model = self.model.cuda()
        self.model.train()

        self.reconstruct()

        local_training_steps = 0
        local_training_num = 0

        while local_training_steps < 3:
            for data in self.dataloader_dict["train"]:
                self.optimizer.zero_grad()
                data = data.cuda()
                pred = self.model(data)
                if "classification" in self.task_type.lower():
                    label = data.y.squeeze(-1).long()
                elif "regression" in self.task_type.lower():
                    label = data.y

                if len(label.size()) == 0:
                    label = label.unsqueeze(0)

                loss = self.criterion(pred, label)

                loss.backward()
                self.optimizer.step()

                local_training_steps += 1
                local_training_num += label.shape[0]
                if local_training_steps>=3:
                    break


        return local_training_num, local_training_steps