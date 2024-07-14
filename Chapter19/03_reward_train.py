#!/usr/bin/env python3
import argparse
import pathlib
import logging
import typing as tt

from lib import rlhf

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, Dataset, ConcatDataset

log = logging.getLogger("reward_train")

SPLIT_SEED = 42
TEST_RATIO = 0.2
INPUT_SHAPE = (3, 210, 160)
TOTAL_ACTIONS = 18
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_EPOCHES = 1000

class LabelsDataset(Dataset):
    def __init__(self, db: rlhf.Database,
                 labels: tt.List[rlhf.HumanLabel],
                 total_actions: int = TOTAL_ACTIONS):
        self.db = db
        self.labels = labels
        self.total_actions = total_actions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # data loading and transformation into tensors
        l = self.labels[idx]
        s1_obs, s1_acts = rlhf.steps_to_tensors(
            self.db.db_root / l.sample1, self.total_actions)
        s2_obs, s2_acts = rlhf.steps_to_tensors(
            self.db.db_root / l.sample2, self.total_actions)

        if l.label == 1:
            vals = [1.0, 0.0]
        elif l.label == 2:
            vals = [0.0, 1.0]
        else:
            vals = [0.5, 0.5]
        mu = torch.as_tensor(vals)
        return s1_obs, s1_acts, s2_obs, s2_acts, mu


def calc_loss(
        model: rlhf.RewardModel,
        s1_obs: torch.ByteTensor, s1_acts: torch.Tensor,
        s2_obs: torch.ByteTensor, s2_acts: torch.Tensor,
        mu: torch.Tensor
) -> torch.Tensor:
    batch_size, steps = s1_obs.size()[:2]

    # combine batch and time sequence dimension into long batch
    s1_obs_flat = s1_obs.flatten(0, 1)
    s1_acts_flat = s1_acts.flatten(0, 1)
    r1_flat = model(s1_obs_flat, s1_acts_flat)
    r1 = r1_flat.view((batch_size, steps))
    R1 = torch.sum(r1, 1)

    s2_obs_flat = s2_obs.flatten(0, 1)
    s2_acts_flat = s2_acts.flatten(0, 1)
    r2_flat = model(s2_obs_flat, s2_acts_flat)
    r2 = r2_flat.view((batch_size, steps))
    R2 = torch.sum(r2, 1)
    R = torch.hstack((R1.unsqueeze(-1), R2.unsqueeze(-1)))
    loss_t = F.binary_cross_entropy_with_logits(R, mu)
    return loss_t


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)s %(message)s", level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-o", "--out", required=True, help="Directory to store the model")
    parser.add_argument("dbs", help="Path to DB", nargs='+')

    args = parser.parse_args()
    print(args)
    device = torch.device(args.dev)

    databases = []
    train_datasets = []
    test_datasets = []
    for db in args.dbs:
        db = rlhf.load_db(db)
        databases.append(db)
        log.info("Loaded DB from %s with %d labels and %d paths",
                 db.db_root, len(db.labels), len(db.paths))
        db.shuffle_labels(SPLIT_SEED)
        pos = int(len(db.labels) * (1-TEST_RATIO))
        train_labels, test_labels = db.labels[:pos], db.labels[pos:]
        train_datasets.append(LabelsDataset(db, train_labels))
        test_datasets.append(LabelsDataset(db, test_labels))
    ds_train = ConcatDataset(train_datasets)
    ds_test = ConcatDataset(test_datasets)
    data_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE)

    writer = SummaryWriter(comment="-reward_" + args.name)
    model = rlhf.RewardModel(INPUT_SHAPE, TOTAL_ACTIONS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)
    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    prev_test_loss = None
    count_overfit = 0
    for epoch in range(MAX_EPOCHES):
        loss = 0.0
        size = 0
        for s1_obs, s1_acts, s2_obs, s2_acts, mu in data_loader:
            optimizer.zero_grad()
            loss_t = calc_loss(
                model, s1_obs, s1_acts, s2_obs, s2_acts, mu
            )
            loss_t.backward()
            optimizer.step()
            loss += loss_t.item()
            size += s1_obs.size()[0]
        train_loss = loss / size

        # test data
        loss = 0.0
        size = 0
        for s1_obs, s1_acts, s2_obs, s2_acts, mu in data_loader_test:
            loss_t = calc_loss(
                model, s1_obs, s1_acts, s2_obs, s2_acts, mu
            )
            loss += loss_t.item()
            size += s1_obs.size()[0]
        test_loss = loss / size

        writer.add_scalar("loss_train", train_loss, epoch)
        writer.add_scalar("loss_test", test_loss, epoch)

        log.info("Epoch %d done, train loss %f, test loss %f",
                 epoch, train_loss, test_loss)
        if prev_test_loss is None or prev_test_loss > test_loss:
            count_overfit = 0
            # save model
            log.info(f"Save model for {test_loss:.5f} test loss")
            path = out_path / ("reward-" + args.name + ".dat")
            torch.save(model.state_dict(), str(path))
            prev_test_loss = test_loss
        else:
            count_overfit += 1
            if count_overfit > 3:
                log.info(f"Prev test loss was less than current for {count_overfit} epoches, stop")
                break
