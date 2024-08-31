import os
import copy
import pickle
import datetime
import time
import torch
from collections.abc import Iterable
from openfgl.utils.basic_utils import total_size

class Logger:
    """Logger class for tracking and saving evaluation results and communication costs.

    Args:
        args (Namespace): Arguments specifying logging and evaluation parameters.
        message_pool (dict): A pool of messages exchanged during the evaluation.
        task_path (str): Path to the task directory.
        personalized (bool): Whether to log personalized communication costs.

    Attributes:
        log_path (str): Path to the log file.
        start_time (float): Start time of the evaluation.
        comm_cost (list): List of communication costs per round.
        metrics_list (list): List of evaluation metrics.
    """
    def __init__(self, args, message_pool, task_path, personalized=False):
        self.args = args
        self.message_pool = message_pool
        self.debug = self.args.debug
        self.task_path = task_path
        self.metrics_list = []
        self.personalized = personalized
        
        
        if args.log_root is None:
            log_root = os.path.join(self.task_path, "debug")
        else:
            log_root = args.log_root
            
        if args.log_name is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
            log_name = f"{self.args.fl_algorithm}_{current_time}.pkl"
        else:
            log_name = args.log_name + ".pkl"
             
        
        self.log_path = os.path.join(log_root, log_name)
        self.start_time = time.time()
        self.comm_cost = []
    
    def add_log(self, evaluation_result):
        """Add an evaluation result to the log.
        Args:
            evaluation_result (dict): The evaluation result to be logged.
        """
        if not self.debug:
            return
        self.metrics_list.append(copy.deepcopy(evaluation_result))
        
        # cost
        if self.args.comm_cost:
            comm_cost = 0
            for client_id in self.message_pool["sampled_clients"]:
                comm_cost += total_size(self.message_pool[f'client_{client_id}'])
            
            if self.personalized:
                comm_cost += total_size(self.message_pool[f'server'])
            else:
                comm_cost += len(self.message_pool['sampled_clients']) * total_size(self.message_pool[f'server'])
            
            self.comm_cost.append(comm_cost)
    
    def save(self):
        """Save the log to a file."""
        if not self.debug:
            return
        
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))
            
            
        log = {
            "args": vars(self.args),
            "time": time.time() - self.start_time,
            "metric": self.metrics_list,
        }
        
        if self.args.comm_cost:
            log["avg_cost_per_round"] = sum(self.comm_cost) / len(self.comm_cost) / 1024 # KB
        with open(self.log_path, 'wb') as file:
            pickle.dump(log, file)
            
"""
fedavg
[client 0] cost: 360.28 KB.
[client 1] cost: 360.28 KB.
[client 2] cost: 360.28 KB.
[client 3] cost: 360.28 KB.
[client 4] cost: 360.28 KB.
[client 5] cost: 360.28 KB.
[client 6] cost: 360.28 KB.
[client 7] cost: 360.28 KB.
[client 8] cost: 360.28 KB.
[client 9] cost: 360.28 KB.
[server] cost: 3602.77 KB.

fedgta
[client 0] cost: 360.83 KB.
[client 1] cost: 360.83 KB.
[client 2] cost: 360.83 KB.
[client 3] cost: 360.83 KB.
[client 4] cost: 360.83 KB.
[client 5] cost: 360.83 KB.
[client 6] cost: 360.83 KB.
[client 7] cost: 360.83 KB.
[client 8] cost: 360.83 KB.
[client 9] cost: 360.83 KB.
[server] cost: 3602.77 KB.

fedtgp/fedproto
[client 0] cost: 1.75 KB.
[client 1] cost: 1.75 KB.
[client 2] cost: 1.75 KB.
[client 3] cost: 1.75 KB.
[client 4] cost: 1.75 KB.
[client 5] cost: 1.75 KB.
[client 6] cost: 1.75 KB.
[client 7] cost: 1.75 KB.
[client 8] cost: 1.75 KB.
[client 9] cost: 1.75 KB.
[server] cost: 17.50 KB.

adafgl
"""
