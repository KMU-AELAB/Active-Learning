import random
import numpy as np

import torch
from torch.backends import cudnn

from config import Config
from query.query_transformer import Query
from query.strategy.strategy_vae import Strategy as autoencoder
from query.strategy.strategy_transformer import Strategy as transformer
from task.classification_loss import ClassificationWithLoss as Task

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_autoencoder(cycle_cnt):
    torch.manual_seed(cycle_cnt * 1000)
    torch.cuda.manual_seed_all(cycle_cnt * 1000)

    config = Config()
    ae = autoencoder(config)

    ae.run()

    return ae


def main(cycle_cnt, config, query, ae):
    random.seed(cycle_cnt * 1000)
    np.random.seed(cycle_cnt * 1000)
    torch.manual_seed(cycle_cnt * 1000)
    torch.cuda.manual_seed_all(cycle_cnt * 1000)

    task = Task(config)
    trans = transformer(config)

    fp = open(f'record_{cycle_cnt}.txt', 'w')
    for step_cnt in range(config.max_cycle):
        # take a new sample
        query.sampling(step_cnt, task, trans, ae)

        # train a task model
        task.run(query.labeled)

        print(f'trial-{cycle_cnt} / step {step_cnt + 1}: train data count - {len(set(query.labeled))}')
        print(f'test accuracy - {task.best_acc}')

        fp.write(f'{task.best_acc}\n')

        trans.run(task, ae, query.labeled)

    fp.close()


if __name__ == '__main__':
    for i in range(5):
        ae = train_autoencoder(i + 1)
        config = Config()
        query = Query(config)

        main(i + 1, config, query, ae)