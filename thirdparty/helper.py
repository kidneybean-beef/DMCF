#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

import numpy as np

precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}

# For TF-TRT:

class OptimizedModel():
    def __init__(self, saved_model_dir = None):
        self.loaded_model_fn = None
        self.call = None
        
        if not saved_model_dir is None:
            self.load_model(saved_model_dir)
            
    
    def predict(self, input_data): 
        if self.loaded_model_fn is None:
            raise(Exception("Haven't loaded a model"))

        # assert len(self.loaded_model_fn.signatures) == 0

        ### CHECK FUNCTION INPUT/OUPUT FORMAT ###
        # test=self.loaded_model_fn.call.get_concrete_function(input_data, training=False)
        # print(test)
        # self.loaded_model_fn.call(input_data,False)
        ### CHECK FUNCTION INPUT/OUPUT FORMAT ###

        return self.loaded_model_fn.call(input_data,False)
    
    def load_model(self, saved_model_dir):
        # wrapper_fp32 = saved_model_loaded.signatures['serving_default']
        # self.loaded_model_fn = wrapper_fp32
        self.loaded_model_fn=tf.saved_model.load(saved_model_dir)
        self.call=self.loaded_model_fn.signatures['serving_default']

class ModelOptimizer():
    def __init__(self, input_saved_model_dir, calibration_data=None):
        self.input_saved_model_dir = input_saved_model_dir
        self.calibration_data = None
        self.loaded_model = None
        self.converter = None
        
        if not calibration_data is None:
            self.set_calibration_data(calibration_data)
        
        
    def set_calibration_data(self, calibration_data):
        
        def calibration_input_fn():
            yield (tf.constant(calibration_data.astype('float32')), )
        
        self.calibration_data = calibration_input_fn
        
        
    def convert(self, output_saved_model_dir=None, precision="FP32"):
        
        if precision == "INT8" and self.calibration_data is None:
            raise(Exception("No calibration data set!"))

        trt_precision = precision_dict[precision]
        conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_precision,
                                                                       use_calibration= precision == "INT8",
                                                                       maximum_cached_engines=28)

        self.converter = tf_trt.TrtGraphConverterV2(input_saved_model_dir=self.input_saved_model_dir,
                                conversion_params=conversion_params, use_dynamic_shape=True,
                                dynamic_shape_profile_strategy="Optimal")
        
        if precision == "INT8":
            self.converter.convert(calibration_input_fn=self.calibration_data)
            self.converter.summary()
        else:
            self.converter.convert()
            self.converter.summary()
            
        self.converter.save(output_saved_model_dir=output_saved_model_dir)
        

        return OptimizedModel(output_saved_model_dir)

    def save(self, output_saved_model_dir):
        self.converter.save(output_saved_model_dir=output_saved_model_dir)

