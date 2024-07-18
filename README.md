# Convert TensorFlow models to Arduino code

This library automates the generation of C++ code to use a TensorFlow model
trained in Python inside an Arduino project.

## Install

```bash
pip install eloquent_tensorflow
```

## How to use

This library is meant to be used in conjunction with the [EloquentTinyML](https://github.com/eloquentarduino/EloquentTinyML)
Arduino library.

Refer to [my blog](https://eloquentarduino.com) for complete tutorials.

```python
from eloquent_tensorflow import convert_model

model = create_and_train_nn_model()
print(convert_model(model))

"""
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
#define TF_NUM_INPUTS 450
#define TF_NUM_OUTPUTS 3
#define TF_NUM_OPS 21

/**
 * Call this function to register the ops
 * that have been detected
 */
template<class TF>
void registerNetworkOps(TF& nn) {
    nn.resolver.AddStridedSlice();
    nn.resolver.AddFill();
    nn.resolver.AddTanh();
    nn.resolver.AddWhile();
    nn.resolver.AddSlice();
    nn.resolver.AddMaximum();
    nn.resolver.AddSoftmax();
    nn.resolver.AddUnidirectionalSequenceLSTM();
    nn.resolver.AddPack();
    nn.resolver.AddGather();
    nn.resolver.AddLess();
    nn.resolver.AddTranspose();
    nn.resolver.AddShape();
    nn.resolver.AddFullyConnected();
    nn.resolver.AddAdd();
    nn.resolver.AddReshape();
    nn.resolver.AddSplit();
    nn.resolver.AddRelu();
    nn.resolver.AddConcatenation();
    nn.resolver.AddMul();
    nn.resolver.AddMinimum();
}

// model data
const unsigned char tfModel[15084] DATA_ALIGN_ATTRIBUTE = { ... };
"""
```