from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.layers import base
from tensorflow.python.layers import utils

class Dense(base.Layer):
  """Densely-connected layer class.
  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).
  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.
  Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the weight matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the weight matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Dense, self).__init__(trainable=trainable, name=name, **kwargs)
    if not isinstance(units, list):
      units = [units]
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.input_spec = base.InputSpec(min_ndim=2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    
    self.kernel = [self.add_variable('kernel_' + str(i),
                                    shape=[input_shape[-1].value if i == 0 else self.units[i-1], num_units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    dtype=self.dtype,
                                    trainable=True) for i, num_units in enumerate(self.units)]
    if self.use_bias:
      self.bias = [self.add_variable('bias_' + str(i),
                                    shape=[num_units,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    dtype=self.dtype,
                                    trainable=True) for i, num_units in enumerate(self.units)]
    else:
      self.bias = None

    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    for i in range(len(self.units)):
      shape = inputs.get_shape().as_list()
      output_shape = shape[:-1] + [self.units[i]]
      if len(output_shape) > 2:
        # Broadcasting is required for the inputs.
        outputs = standard_ops.tensordot(inputs, self.kernel[i], [[len(shape) - 1],
                                                               [0]])
        # Reshape the output back to the original ndim of the input.
        outputs.set_shape(output_shape)
      else:
        outputs = standard_ops.matmul(inputs, self.kernel[i])
      if self.use_bias:
        outputs = nn.bias_add(outputs, self.bias[i])
      if self.activation is not None:
        return self.activation(outputs)  # pylint: disable=not-callable

      inputs = outputs

    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units[-1])

