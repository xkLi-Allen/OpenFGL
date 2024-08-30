import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient

class IsolateClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(IsolateClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        
    def execute(self):
        self.task.train()

    def send_message(self):
        pass
        
