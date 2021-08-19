# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle

from net import BenchmarkDNNLayer
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()
        self.sync_mode = config.get("runner.sync_mode")
        self.optimizer = None

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.dict_dim = self.config.get("hyper_parameters.dict_dim")
        self.emb_dim = self.config.get("hyper_parameters.emb_dim")
        self.slot_num = self.config.get("hyper_parameters.slot_num")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.layer_sizes = self.config.get("hyper_parameters.layer_sizes")

    def create_feeds(self, is_infer=False):

        slot_ids = [
            paddle.static.data(
                name=str(i), shape=[None, 1], dtype="int64", lod_level=1)
            for i in range(2, self.slot_num + 2)
        ]

        label = paddle.static.data(
            name="1", shape=[None, 1], dtype="int64", lod_level=1)

        feeds_list = [label] + slot_ids
        return feeds_list

    def net(self, input, is_infer=False):
        self.label_input = input[0]
        self.slot_inputs = input[1:]

        dnn_model = BenchmarkDNNLayer(
            self.dict_dim,
            self.emb_dim,
            self.slot_num,
            self.layer_sizes,
            sync_mode=self.sync_mode)

        self.predict = dnn_model(self.slot_inputs)

        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)
        #label_int = paddle.cast(self.label, 'int64')
        auc, batch_auc_var, _ = paddle.static.auc(input=predict_2d,
                                                  label=self.label_input,
                                                  slide_steps=0)
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict
        cost = paddle.nn.functional.log_loss(
            input=self.predict, label=paddle.cast(self.label_input, "float32"))
        avg_cost = paddle.sum(x=cost)
        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        # if strategy != None:
        #     import paddle.distributed.fleet as fleet
        #     optimizer = fleet.distributed_optimizer(optimizer, strategy)
    
        optimizer = paddle.static.amp.bf16.decorate_bf16(
            optimizer, amp_lists=paddle.static.amp.bf16.AutoMixedPrecisionListsBF16(
                custom_bf16_list={'matmul, elementwise_add, scale, sigmoid'}, ), use_bf16_guard=False, use_pure_bf16=False)
        logger.info("WARNING!!!! Here it means bf16 is set")
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
