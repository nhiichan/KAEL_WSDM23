import logging
import math
import os

import torch

from BaseDetectors import params
from BaseDetectors.ComplEx import ComplEx
from BaseDetectors.DistMult import DistMult
from BaseDetectors.TransE import TransE
from BaseDetectors.create_batch import get_batch_baseline
from BaseDetectors.dataset import Reader


def base_detector(model_name, ratio, num_num):
    # params.num_anomaly_num = ratio
    params.num_anomaly_num = params.num_anomaly_num * ratio
    params.kkkkk = num_num
    data_path = params.data_dir_umls
    data_name = "UMLS"
    # data_name = "WN18RR"
    dataset = Reader(data_path, "train", isInjectTopK=True)
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(
            params.log_folder,
            "{0}_{1}_{2}_{3}_log.txt".format(model_name, data_name, str(math.floor(params.num_anomaly_num)),
                                             str(params.kkkkk))))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    # global dataset, params
    print("Model name:", model_name)
    logger.info('============ Initialized logger ============')
    logger.info('============================================')
    logging.info('There are %d Triples with %d anomalies in the graph.' %
                 (len(dataset.labels), math.floor(params.num_anomaly_num)))

    params.total_ent = dataset.num_entity
    params.total_rel = dataset.num_relation
    # Model
    if model_name == "DistMult":
        model = DistMult(params)
    elif model_name == "ComplEx":
        model = ComplEx()
    elif model_name == "TransE":
        model = TransE()
    else:
        logging.info('No such a model!!!')
        exit()
    model = model.to(params.device)

    model_saved_path = model_name + "_" + data_name + "_" + str(math.floor(
        params.num_anomaly_num)) + ".ckpt"

    logging.info(model_saved_path)
    model_saved_path = os.path.join(params.out_folder, model_saved_path)
    # model.load_state_dict(torch.load(os.path.join(params.out_folder, "TransE_FB_model_0.05_2740.ckpt")))
    # criterion = nn.MarginRankingLoss(params.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    all_triples = dataset.train_data
    # print(all_triples)
    labels = dataset.labels
    # print(labels)

    triplelist = dataset.triplelist

    train_idx = list(range(len(all_triples) // 2))

    # print("length of idx & train data", len(train_idx), len(all_labels))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies /
                               params.batch_size)
    for k in range(params.kkkkk):
        for epoch in range(num_iterations):
            batch_h, batch_t, batch_r, batch_y = get_batch_baseline(
                all_triples, train_idx, epoch)

            batch_h_tensor = torch.LongTensor(batch_h).to(params.device)
            batch_t_tensor = torch.LongTensor(batch_t).to(params.device)
            batch_r_tensor = torch.LongTensor(batch_r).to(params.device)
            # print('batch_h_tensor', batch_h_tensor)

            loss, pos_score, neg_score, scoredict = model(
                batch_h_tensor, batch_t_tensor, batch_r_tensor, batch_y)
            # print(scoredict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'Epoch %d--%d with loss: %f, positive_loss: %f, negative loss: %f'
                % (k, epoch, loss, pos_score[0].cpu().data,
                   neg_score[0].cpu().data))
            logging.info('%d %d' % (len(batch_h_tensor), len(scoredict)))

            torch.save(model.state_dict(), model_saved_path)
    # print(len(triplelist))
    # print(triplelist)
    # print("HELLO", scoredict.get('[86, 28, 102]'))
    # print(triplelist['[86, 28, 102]'])
    return scoredict, triplelist
