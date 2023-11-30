Custom Architectures
=========================

.. code-block:: python

    from torch import nn

    from antelope import ml

    class Model(nn.Module):
        def __init__(self, in_features, out_features):

            super().__init__()

            self.MLP = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features, 32),
                                     nn.Linear(32, out_features))

        def forward(self, x):
            return self.MLP(x)

    ml(model=Model, dataset='CIFAR10')

^^^^^^^^^^^^^^^^^^^
Input and output shapes
^^^^^^^^^^^^^^^^^^^

antelope-brethren-star automatically detects the shape signature of your model.

.. code:: diff

   class Model(nn.Module):
   +   def __init__(self, in_features, out_features):
           super().__init__()

           self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

       def forward(self, x):
           return self.model(x)


Inferrable signature arguments include ``in_shape``, ``out_shape``,
``in_features``, ``out_features``, ``in_channels``, ``out_channels``,
``in_dim``, ``out_dim``.

Just include them as args to your model and The Antelope will detect and
fill them in.

^^^^^^^^^^^^^^^^^^^
Shaping pre-computation
^^^^^^^^^^^^^^^^^^^

Architectures can also be passed in piece-wise using the piece-wise syntax. For example, ``eyes=ResNet18 predictor=MLP`` will combine the encoder of ResNet18 with an MLP predictor head. In this case, each piece, or "part", pre-computes the output shape of the earlier. To make this shape pre-computation faster, and define it yourself, you can give your architecture a ``shape(self, in_shape):`` method that outputs the expected output shape given an input shape.

It's best practice to always include this method just to simplify shape pre-computation.

^^^^^^^^^^^^^^^^^^^
Example: ResNet18
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from antelope.Agents.Blocks.Architectures.Residual import Residual

    from antelope.Utils import cnn_feature_shape


    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, down_sample=None):
            super().__init__()

            if down_sample is None and (in_channels != out_channels or stride != 1):
                down_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                      kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channels))

            block = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                            kernel_size=kernel_size, padding=kernel_size // 2,
                                            stride=stride, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels,
                                            kernel_size=kernel_size, padding=kernel_size // 2,
                                            bias=False),
                                  nn.BatchNorm2d(out_channels))

            self.ResBlock = nn.Sequential(Residual(block, down_sample),
                                          nn.ReLU(inplace=True))

        def shape(self, in_shape):
            return cnn_feature_shape(in_shape, self.ResBlock)  # Pre-computes the shapes of basic CNNs

        def forward(self, x):
            return self.ResBlock(x)


    class ResNet18(nn.Module):
        """
        A full ResNet backbone with computationally-efficient defaults.
        """
        def __init__(self, in_channels, kernel_size=3, stride=2, dims=(64, 64, 128, 256, 512), depths=(2, 2, 2, 2)):
            super().__init__()

            # ResNet
            self.ResNet = nn.Sequential(nn.Conv2d(in_channels, dims[0],
                                                  kernel_size=kernel_size, padding=1, bias=False),
                                        nn.BatchNorm2d(dims[0]),
                                        nn.ReLU(inplace=True),
                                        *[nn.Sequential(*[ResBlock(dims[i + (j > 0)], dims[i + 1], kernel_size,
                                                                   1 + (stride - 1) * (i > 0 and j > 0))
                                                          for j in range(depth)])
                                          for i, depth in enumerate(depths)])

        def shape(self, in_shape):
            return cnn_feature_shape(in_shape, self.ResNet)  # Pre-computes the shapes of basic CNNs

        def forward(self, x):
            return self.ResNet(x)


    ml(model=ResNet18, dataset='CIFAR10')


^^^^^^^^^^^^^^^^^^^
Example: TIMM
^^^^^^^^^^^^^^^^^^^

You have access to all the computer vision models of TIMM, straight away!

For example, MobileNet:

.. code-block:: python

    from antelope.Agents.Blocks.Architectures.TIMM import TIMM

    ml(model=TIMM(name='mobilenetv2_100.ra_in1k'))


All TIMM models are listed `here <https://huggingface.co/timm>`_.

^^^^^^^^^^^^^^^^^^^
CNN
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Model(nn.Module):
        def __init__(self, in_channels):
            super().__init__()

            self.CNN = nn.Sequential(nn.Conv2d(in_channels, 8), nn.MaxPool2d(2, 2),
                                     nn.Conv2d(8, 8))

        def forward(self, x):
            return self.CNN(x)

    ml(model=Model, dataset='CIFAR10')

Since there are no args for the out-shape (e.g. ``out_shape``, ``out_channels``, ``out_features``), the convolution will be followed by a default flattening and output mapping neural network. However, including a ``learn`` method would disable this automatic extrapolation. Once a ``learn`` method is added, the model is considered sufficiently advanced that it doesn't need to be automatically extrapolated.

^^^^^^^^^^^^^^^^^^^
If you’re feeling brave, this works as well:
^^^^^^^^^^^^^^^^^^^

Not exactly scalable, but:

.. code:: console

   ml model='nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))' dataset=CIFAR10