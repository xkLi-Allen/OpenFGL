import torch
from torch.utils.data import Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from openfgl.flcore.feddep.dec_cluster.ptdec.dec import DEC
from openfgl.flcore.feddep.dec_cluster.ptdec.model import train, predict
from openfgl.flcore.feddep.dec_cluster.ptsdae.sdae import StackedDenoisingAutoEncoder
import openfgl.flcore.feddep.dec_cluster.ptsdae.model as ae


class Node_Emb_Dateset(Dataset):
    def __init__(self, node_embs, testing_mode=False):
        if isinstance(node_embs, torch.Tensor) == False:
            self.node_embs = torch.Tensor(node_embs)
        else:
            self.node_embs = node_embs
        self.testing_mode = testing_mode

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.node_embs[index]

    def __len__(self) -> int:
        return len(self.node_embs)


def train_clustering(
    node_embs,
    num_prototypes,
    batch_size,
    device,
    ae_pretrained_epochs,
    ae_finetune_epochs,
    dec_epochs,
):
    ds_train = Node_Emb_Dateset(
        node_embs=node_embs, testing_mode=False
    )  # training dataset
    emb_len = len(node_embs[0])
    autoencoder = StackedDenoisingAutoEncoder(
        [emb_len, 128, 512, 128, emb_len], final_activation=None
    )
    
    autoencoder.to(device)
    print("Pretraining stage.")
    ae.pretrain(
        ds_train,
        autoencoder,
        device=device,
        validation=None,
        epochs=ae_pretrained_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
    )
    # print("Training stage.")
    ds_train = Node_Emb_Dateset(node_embs=node_embs, testing_mode=False)
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.01, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        device=device,
        validation=None,
        epochs=ae_finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
    )
    print("DEC stage.")
    model = DEC(
        cluster_number=num_prototypes,
        hidden_dimension=emb_len,
        encoder=autoencoder.encoder,
    )
    model.to(device)
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=dec_epochs,
        batch_size=batch_size,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        device=device
    )
    predicted = predict(ds_train, model, 1024, silent=True, device=device)
    predicted = predicted.cpu().numpy()
    print("Finish DEC!")

    return predicted
