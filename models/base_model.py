import numpy as np
import yaml
import tensorflow as tf
from os.path import join, exists, dirname, abspath
from abc import ABC, abstractmethod

from o3d.utils import Config


class BaseModel(ABC, tf.keras.Model):
    """Base class for models.

    All models must inherit from this class and implement all functions to be
    used with a pipeline.

    Args:
        **kwargs: Configuration of the model as keyword arguments.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.cfg = Config(kwargs)

    # to save the call function in the saved model, it has to be declared as '@tf.function'

    # def call(self, data, training=True, **kwargs):
    @tf.function(input_signature=(
         tf.TensorSpec(shape=(None,3), dtype=tf.float32),
         tf.TensorSpec(shape=(None,3), dtype=tf.float32),
         tf.TensorSpec(shape=(None,3), dtype=tf.float32),
         tf.TensorSpec(shape=None, dtype=tf.float32),
         tf.TensorSpec(shape=(None,3), dtype=tf.float32),
         tf.TensorSpec(shape=(None,3), dtype=tf.float32),
         tf.TensorSpec(shape=(),dtype=tf.bool)),reduce_retracing=True)
    def call(self, pos, vel, acc, feats, box, bfeats, training=False, **kwargs):
        # data = [pos, vel, acc, feats, box, bfeats]

        # d = self.transform(data, training=training, **kwargs)
        # x = self.preprocess(d, training=training, **kwargs)
        # x = self.forward(x, d, training=training, **kwargs)
        # x = self.postprocess(x, d, training=training, **kwargs)
        # x = self.inv_transform(x, data, training=training, **kwargs)

        # pos_trans, vel_trans, acc_trans, feats_trans, box_trans, bfeats_trans = self.transform(data, training=training, **kwargs)
        # d = [pos_trans, vel_trans, acc_trans, feats_trans, box_trans, bfeats_trans]
        # dilated_pos, fluid_feats, idx, dens = self.preprocess(d, training=training, **kwargs)
        # x = [dilated_pos, fluid_feats, idx, dens]
        # dilated_pos, fluid_feats, idx, dens = self.forward(x, d, training=training, **kwargs)
        # x = [dilated_pos, fluid_feats, idx, dens]
        # pos2_corrected, vel2_corrected = self.postprocess(x, d, training=training, **kwargs)
        # x = [pos2_corrected, vel2_corrected]
        # x = self.inv_transform(x, data, training=training, **kwargs)

        pos_trans, vel_trans, acc_trans, feats_trans, box_trans, bfeats_trans = self.transform(pos, vel, acc, feats, box, bfeats, training=training, **kwargs)
        # d = [pos_trans, vel_trans, acc_trans, feats_trans, box_trans, bfeats_trans]
        dilated_pos, fluid_feats, idx, dens = self.preprocess(pos_trans, vel_trans, acc_trans, feats_trans, box_trans, bfeats_trans, training=training, **kwargs)
        # x = [dilated_pos, fluid_feats, idx, dens]
        pos_fwd = self.forward(dilated_pos, fluid_feats, idx, dens, pos_trans, vel_trans, acc_trans, feats_trans, box_trans, bfeats_trans, training=training, **kwargs)
        # x = [dilated_pos, fluid_feats, idx, dens]
        pos2_corrected, vel2_corrected = self.postprocess(pos_fwd, pos_trans, vel_trans, acc_trans, training=training, **kwargs)
        # x = [pos2_corrected, vel2_corrected]
        x = self.inv_transform(pos2_corrected, vel2_corrected, pos, vel, acc, feats, box, bfeats, training=training, **kwargs)
        # x = self.inv_transform(x, data, training=training, **kwargs)
        return x
        # return x['output_1'], x['output_2']

    @abstractmethod
    def forward(self, prev, data, training=True, **kwargs):
        return

    @abstractmethod
    def loss(self, results, data):
        """Computes the loss given the network input and outputs.

        Args:
            results: This is the output of the model.
            inputs: This is the input to the model.

        Returns:
            Returns the loss value.
        """
        return {}

    @abstractmethod
    def get_optimizer(self, cfg_pipeline):
        """Returns an optimizer object for the model.

        Args:
            cfg_pipeline: A Config object with the configuration of the pipeline.

        Returns:
            Returns a new optimizer object.
        """

        return

    def transform(self, data, training=True, **kwargs):
    # def transform(self, pos, vel, acc, feats, box, bfeats, training=True, **kwargs):
        """Transformation step.

        Args:
            input: Input of model

        Returns:
            Returns modified input.
        """

        return input

    def inv_transform(self, prev, data, training=True, **kwargs):
        """Inversion transformation step.

        Args:
            input: Output of model

        Returns:
            Returns modified output.
        """

        return input

    def preprocess(self, data, training=True, **kwargs):
        """Preprocessing step.

        Args:
            input: Input of model

        Returns:
            Returns modified input.
        """

        return input

    def postprocess(self, prev, data, training=True, **kwargs):
        """Preprocessing step.

        Args:
            input: Output of model

        Returns:
            Returns modified output.
        """

        return input
