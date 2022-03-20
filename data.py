# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re

import paddle
import numpy as np
import pandas as pd

from paddlenlp.datasets import MapDataset


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            yield {'query': data[0], 'title': data[1]}


def read_excel_pair(data_path, is_test=False):
    """Reads data."""
    data = pd.read_excel(data_path)
    for index, line in data.iterrows():
        query = clean_text(str(line['query']))
        title = clean_text(str(line['title']))
        if is_test:
            yield {'query': query, 'title': title}
        else:
            yield {'query': query, 'title': title, 'label': line['label']}


def tm_ind_fire_ds_realtime(fire_data, ind_data):
    """
    Text Matching
    Using a fire supervision system combined text to match all industrial combined text.
    Instead of output file, output paddle dataset.

    :param fire_data:
    :param ind_data:
    :return:
    """

    for index_fire, fire in fire_data.iterrows():
        for index_ind, ind in ind_data.iterrows():
            yield {'query': ind['qymc'], 'title': fire['basic_com_info_xfjd.dwmc']}


def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text


def write_excel_results(filename, results):
    """write test results"""
    df = pd.read_excel(filename)

    df['result'] = [x[0] for x in results]
    with pd.ExcelWriter('sample_test.xlsx') as writer:
        df.to_excel(writer, index=False)
    print("Test results saved")


def tm_select_top_same_excel(fire_data, ind_data, results):
    """
    Text Matching
    Write test results

    :param fire_data:
    :param ind_data:
    :param results:
    :return:
    """

    df = pd.DataFrame(columns=['fire_id', 'ind_guid', 'query', 'title'])
    for index_fire, fire in fire_data.iterrows():
        dataset = pd.DataFrame(columns=['fire_id', 'ind_guid', 'query', 'title'])
        dataset['ind_guid'] = ind_data['guid']
        dataset['query'] = ind_data['qymc']
        dataset['fire_id'] = fire['basic_com_info_xfjd.id']
        dataset['title'] = fire['basic_com_info_xfjd.dwmc']

        df = pd.concat([df, dataset], axis=0, ignore_index=True)

    df['result'] = [x[0] for x in results]

    pre_group = df.groupby([
        'fire_id'
    ], sort=False)['result'].nlargest(5)

    pre_idx = pre_group.reset_index(level='fire_id', drop=True).index

    filename = 'test-timeCNAddrTeam.xlsx'
    if os.path.exists(filename):
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.iloc[pre_idx].to_excel(writer, header=False, index=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        with pd.ExcelWriter(filename, mode='w') as writer:
            df.iloc[pre_idx].to_excel(writer, index=False)
    print("Test results saved")


def tm_select_top_same_excel_split(fire_data, ind_data, results):
    """
    Text Matching
    Write test results split

    :param fire_data:
    :param ind_data:
    :param results:
    :return:
    """
    ind_len = len(ind_data)
    filename = 'test-timeCNAddrTeam.xlsx'
    for index_fire, fire in fire_data.iterrows():
        dataset = pd.DataFrame(columns=['fire_id', 'ind_guid', 'query', 'title', 'result'])
        dataset['ind_guid'] = ind_data['guid']
        dataset['query'] = ind_data['qymc']
        dataset['fire_id'] = fire['basic_com_info_xfjd.id']
        dataset['title'] = fire['basic_com_info_xfjd.dwmc']
        dataset['result'] = [x[0] for x in results[index_fire * ind_len:(index_fire + 1) * ind_len]]

        pre_group = dataset.nlargest(5, 'result')
        if os.path.exists(filename):
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                pre_group.to_excel(writer, header=False, index=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            with pd.ExcelWriter(filename, mode='w') as writer:
                pre_group.to_excel(writer, index=False)


def convert_pointwise_example(example,
                              tokenizer,
                              max_seq_length=512,
                              is_test=False):
    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def convert_pairwise_example(example,
                             tokenizer,
                             max_seq_length=512,
                             phase="train"):
    if phase == "train":
        query, pos_title, neg_title = example["query"], example[
            "title"], example["neg_title"]

        pos_inputs = tokenizer(
            text=query, text_pair=pos_title, max_seq_len=max_seq_length)
        neg_inputs = tokenizer(
            text=query, text_pair=neg_title, max_seq_len=max_seq_length)

        pos_input_ids = pos_inputs["input_ids"]
        pos_token_type_ids = pos_inputs["token_type_ids"]
        neg_input_ids = neg_inputs["input_ids"]
        neg_token_type_ids = neg_inputs["token_type_ids"]

        return (pos_input_ids, pos_token_type_ids, neg_input_ids,
                neg_token_type_ids)

    else:
        query, title = example["query"], example["title"]

        inputs = tokenizer(
            text=query, text_pair=title, max_seq_len=max_seq_length)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        if phase == "eval":
            return input_ids, token_type_ids, example["label"]
        elif phase == "predict":
            return input_ids, token_type_ids
        else:
            raise ValueError("not supported phase:{}".format(phase))


def gen_pair(dataset, pool_size=100):
    """
    Generate triplet randomly based on dataset

    Args:
        dataset: A `MapDataset` or `IterDataset` or a tuple of those. 
            Each example is composed of 2 texts: exampe["query"], example["title"]
        pool_size: the number of example to sample negative example randomly

    Return:
        dataset: A `MapDataset` or `IterDataset` or a tuple of those.
        Each example is composed of 2 texts: exampe["query"], example["pos_title"]、example["neg_title"]
    """

    if len(dataset) < pool_size:
        pool_size = len(dataset)

    new_examples = []
    pool = []
    tmp_exmaples = []

    for example in dataset:
        label = example["label"]

        # Filter negative example
        if label == 0:
            continue

        tmp_exmaples.append(example)
        pool.append(example["title"])

        if len(pool) >= pool_size:
            np.random.shuffle(pool)
            for idx, example in enumerate(tmp_exmaples):
                example["neg_title"] = pool[idx]
                new_examples.append(example)
            tmp_exmaples = []
            pool = []
        else:
            continue
    return MapDataset(new_examples)
