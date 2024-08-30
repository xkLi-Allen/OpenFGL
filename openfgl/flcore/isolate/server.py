import torch
from openfgl.flcore.base import BaseServer

class IsolateServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(IsolateServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        
   
    def execute(self):
        assert len(self.message_pool["sampled_clients"]) == self.args.num_clients
        pass
        
    def send_message(self):
        pass