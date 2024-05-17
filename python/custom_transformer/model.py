# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
from typing import Dict, Union

from kserve import Model, ModelServer, model_server, InferInput, InferRequest, InferResponse
from kserve.model import PredictorProtocol, PredictorConfig


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str, predictor_protocol: str, predictor_use_ssl: bool):
        super().__init__(name, PredictorConfig(predictor_host, predictor_protocol, predictor_use_ssl))
        self.ready = True

    def preprocess(self, payload, headers: Dict[str, str] = None) \
            -> Union[Dict, InferRequest]:

        print("--------req payload----------")
        print(payload)
        print("--------req payload----------")
        image_64_encode = base64.b64encode(payload)
        bytes_array = image_64_encode.decode('utf-8')
        # Transform to KServe v1/v2 inference protocol
        if self.protocol == PredictorProtocol.REST_V1.value:
            inputs = [{"data": bytes_array}]
            payload = {"instances": inputs}
            print("--------trans result----------")
            print(payload)
            print("--------trans result----------")
            return payload
        else:
            print("--------other Protocol----------")
            return


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = ImageTransformer(args.model_name, predictor_host=args.predictor_host,
                             predictor_protocol=args.predictor_protocol,
                             predictor_use_ssl=args.predictor_use_ssl)
    ModelServer().start([model])
