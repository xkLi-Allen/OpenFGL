import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm
from openfgl.flcore.feddep.dec_cluster.ptdec.utils import target_distribution


def train(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    stopping_delta: Optional[float] = None,
    collate_fn=default_collate,
    device=None,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: int = 10,
    evaluate_batch_size: int = 1024,
    update_callback: Optional[Callable[[float, float], None]] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=True,
    )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
        },
        disable=silent,
    )
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        batch = batch.to(device)
        features.append(model.encoder(batch).detach().cpu())
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(
        np.copy(predicted), dtype=torch.long
    )  # [0,1,2,4,2,4,...]
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
    cluster_centers = cluster_centers.to(device)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    loss_function = nn.KLDivLoss(reduction='sum')
    delta_label = None
    for epoch in range(epochs):
        features = []
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % (delta_label or 0.0),
            },
            disable=silent,
        )
        model.train()
        for index, batch in enumerate(data_iterator):
            if not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch)
            batch = batch.to(device)
            output = model(batch)
            target = target_distribution(output).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo=epoch,
                lss="%.8f" % float(loss.item()),
                dlb="%.4f" % (delta_label or 0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            features.append(model.encoder(batch).detach().cpu())
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.8f" % loss_value,
                    dlb="%.4f" % (delta_label or 0.0),
                )
                if update_callback is not None:
                    update_callback(loss_value, delta_label)
        predicted = predict(
            dataset,
            model,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            device=device
        )
        delta_label = (
            float((predicted != predicted_previous).float().sum().item())
            / predicted_previous.shape[0]
        )
        if stopping_delta is not None and delta_label < stopping_delta:
            print(
                'Early stopping as label delta "%1.5f" less than "%1.5f".'
                % (delta_label, stopping_delta)
            )
            break
        predicted_previous = predicted
        data_iterator.set_postfix(
            epo=epoch,
            lss="%.8f" % 0.0,
            dlb="%.4f" % (delta_label or 0.0),
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model)


def predict(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    device=None,
    silent: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    data_iterator = tqdm(
        dataloader,
        leave=True,
        unit="batch",
        disable=silent,
    )
    features = []
    model.eval()
    for batch in data_iterator:
        batch = batch.to(device)
        features.append(
            model(batch).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    res = torch.cat(features)
    return torch.argmax(res, dim=1)
