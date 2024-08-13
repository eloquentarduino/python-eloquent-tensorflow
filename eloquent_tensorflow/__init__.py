"""
Requires Keras 2.x
Doesn't work with Keras 3.x
"""
import hexdump
import numpy as np
import tensorflow as tf
from tempfile import TemporaryDirectory
from jinja2 import Template
from tensorflow.lite.python.convert_phase import ConverterError


TEMPLATE = """
#pragma once

#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

// automatically configure network
{% if num_inputs is not none %}#define TF_NUM_INPUTS {{ num_inputs }}{% endif %}
{% if num_outputs is not none %}#define TF_NUM_OUTPUTS {{ num_outputs }}{% endif %}
#define TF_NUM_OPS {{ allowed_layers | length }}

{% if allowed_layers | length > 0 %}/**
 * Call this function to register the ops
 * that have been detected
 */
template<class TF>
void registerNetworkOps(TF& nn) {
    {% for layer in allowed_layers %}nn.resolver.Add{{ layer }}();
    {% endfor %}
}
{% endif %}

{% if not_allowed_layers | length %}// these layers are used in Python
// but are not allowed in Arduino
{% for layer in not_allowed_layers %}// - {{ layer }}
{% endfor %}{% endif %}

// model data
const unsigned char {{ model_name }}[{{ model_size }}] DATA_ALIGN_ATTRIBUTE = { {{ bytes_array }} };
"""


def convert_model(model, X: np.ndarray = None, y: np.ndarray = None, model_name: str = 'tfModel') -> str:
    """
    Convert TensorFlow model to C header for Arduino
    :param model:
    :param X:
    :param y:
    :param model_name:
    :return:
    """
    input_shape = [d or 1 for d in model.layers[0].input_shape]
    num_inputs = np.prod(input_shape[1:])
    num_outputs = model.layers[-1].output_shape[1]

    # give user hint of which layers to include
    unique_layers = set([layer.__class__.__name__ for layer in model.layers])
    layer_mapping = {
        'Add': 'Add',
         'AvgPool2D': 'AveragePool2D',
         'Concatenate': 'Concatenation',
         'Conv2D': 'Conv2D',
         'Dense': 'FullyConnected',
         'DepthwiseConv2D': 'DepthwiseConv2D',
         'ELU': 'Elu',
         'Flatten': 'Reshape',
         'LSTM': 'UnidirectionalSequenceLSTM',
         'LeakyReLU': 'LeakyRelu',
         'MaxPool2D': 'MaxPool2D',
         'MaxPooling2D': 'MaxPool2D',
         'Maximum': 'Maximum',
         'Minimum': 'Minimum',
         'PReLU': 'Prelu',
         'ReLU': 'Relu',
         'Reshape': 'Reshape',
         'Softmax': 'Softmax'
     }

    dependencies = {
        'LSTM': ['Shape', 'Reshape', 'StridedSlice', 'Pack', 'Fill', 'Transpose', 'While', 'Less', 'Add', 'Gather', 'Split', 'Mul', 'Minimum', 'Maximum', 'Relu', 'Tanh', 'Concatenation', 'Slice']
    }

    # detect allowed and not allowed layers
    allowed_layers = ['Softmax']
    not_allowed_layers = []

    for layer in unique_layers:
        if layer not in layer_mapping:
            not_allowed_layers.append(layer)
            continue

        allowed_layers += [layer_mapping[layer]] + dependencies.get(layer, [])

    allowed_layers = set(allowed_layers)
    not_allowed_layers = set(not_allowed_layers)

    # convert model to bytes
    if 'UnidirectionalSequenceLSTM' in allowed_layers:
        # see https://github.com/tensorflow/tflite-micro/issues/2006#issuecomment-1567349993
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(tf.TensorSpec(input_shape, model.inputs[0].dtype))

        with TemporaryDirectory() as model_dir:
            model.save(model_dir, save_format='tf', signatures=concrete_func)
            converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
            converted = converter.convert()
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        try:
            converted = converter.convert()
        except ConverterError:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converted = converter.convert()

    model_bytes = hexdump.dump(converted).split(' ')
    bytes_array = ', '.join(['0x%02x' % int(byte, 16) for byte in model_bytes])
    model_size = len(model_bytes)

    # use Jinja to generate clean code
    return Template(TEMPLATE).render(**locals())
