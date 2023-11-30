Original
#############

Welcome
=======

See our library
`Tributaries <../../../tributaries-ml/src/tributaries>`__ for
mass-deploying UnifiedML apps on remote servers.

Check out `minihydra / leviathan <../../../minihydra/src/minihydra>`__ for how we handle sys args &
hyperparams.

Install
-------

.. code:: console

   pip install UnifiedML

What is UnifiedML?
==================

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

UnifiedML is a toolbox & engine for defining ML tasks and training them
individually, or together in a single general intelligence.

Basics
======

Training example
----------------

The default domain is classification.

Train a two-layer neural network on the CIFAR10 classification dataset:

.. code:: python

   # Run.py

   from torch import nn

   model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))  # Two-layer neural-net

**Run:**

.. code:: console

   ML Model=Run.model Dataset=CIFAR10

There are many `built-in <#built-ins>`__ datasets, architectures,
environments, and so on, such as ``Dataset=CIFAR10``. The domain can be
changed with ``task=`` `as we’ll see later <#syntax>`__. `Custom
datasets <#tutorials>`__ and much more can be passed in with analogous
syntax (e.g. ``Dataset=``, ``Env=``, etc.).

show side by side results plots

Search paths
------------

The earlier demonstrates **dot notation** (``Run.model``) for pointing
to an ML object. Equivalently, it’s possible to use **regular directory
paths**:

.. code:: console

   ML Model=./Run.py.model Dataset=CIFAR10

Wherever you run ``ML``, it’ll search from the current directory for any
specified paths.

If you’re feeling brave, this works as well:
--------------------------------------------

Not exactly scalable, but:

.. code:: console

   ML Model='nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))' Dataset=CIFAR10

Custom models
-------------

.. code:: python

   # Run.py

   from torch import nn

   class Model(nn.Module):
      def __init__(self, in_features, out_features):
           super().__init__()

           self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

       def forward(self, x):
           return self.model(x)

**Run:**

.. code:: console

   ML Model=Run.Model Dataset=CIFAR10

Apps
----

It’s possible to do this entirely from code without using ``ML``, as per
below:

.. code:: python

   # Run.py

   from torch import nn

   from ML import main

   model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))

   if __name__ == '__main__':
       main(Model=model, Dataset='CIFAR10')

**Run:**

.. code:: console

   # Equivalent pure-code training example

   python Run.py

We call this a UnifiedML **app**.

Syntax
------

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h3>

Here’s how to write the same program in 7 different ways. (Click to
expand)

.. raw:: html

   </h3>

.. raw:: html

   </summary>

Train a simple 5-layer CNN to play Atari Pong:

Way 1. Purely command-line
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   ML task=RL Env=Atari env.game=pong Model=CNN model.depth=5

Way 2. Command-line code
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   ML task=RL Env='Atari(game="pong")' Model='CNN(depth=5)'

Way 3. Command-line
~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Run.py

   from ML import main

   if __name__ == '__main__':
       main()

**Run:**

.. code:: console

   python Run.py task=RL Env=Atari env.game=pong Model=CNN model.depth=5

Way 4. Inferred Code
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Run.py

   from ML import main

   if __name__ == '__main__':
       main('env.game=pong', 'model.depth=5', task='RL', Env='Atari', Model='CNN')

**Run:**

.. code:: console

   python Run.py

Way 5. Purely Code
~~~~~~~~~~~~~~~~~~

.. code:: python

   # Run.py

   from ML import main
   from ML.Blocks.Architectures import CNN
   from ML.World.Environments import Atari

   if __name__ == '__main__':
       main(task='RL', Env=Atari(game='pong'), Model=CNN(depth=5))

**Run:**

.. code:: console

   python Run.py

Way 6. Recipes
~~~~~~~~~~~~~~

Define recipes in a ``.yaml`` file like this one:

.. code:: yaml

   # recipe.yaml

   imports:
     - RL
     - self
   Env: Atari
   env:
     game: pong
   Model: CNN
   model:
     depth: 5

**Run:**

.. code:: console

   ML task=recipe

The ``imports:`` syntax allows importing multiple tasks/recipes from
different sources, with the last item in the list having the highest
priority when arguments conflict.

Custom task ``.yaml`` files will be searched for in the root directory
``./``, a ``Hyperparams/`` directory if one exists, and a
``Hyperparams/task`` directory if one exists.

Way 7. All of the above
~~~~~~~~~~~~~~~~~~~~~~~

The order of hyperparam priority is ``command-line > code > recipe``.

Here’s a combined example:

.. code:: yaml

   # recipe.yaml

   imports:
     - RL
     - self
   Model: CNN
   model:
     depth: 5

.. code:: python

   # Run.py

   from ML import main
   from ML.World.Environments.Atari import Atari

   if __name__ == '__main__':
       main(Env=Atari)

**Run:**

.. code:: console

   python Run.py task=recipe env.game=pong

.. raw:: html

   </details>

--------------

Valid ML objects include not just instantiated objects, but classes as
well.

1. **Class argument tinkering** The ``hyperparam.`` syntax is used to
   modify arguments of flag ``Hyperparam``. We reserve
   ``Uppercase=Path.To.Class`` for the class itself and
   ``lowercase.key=value`` for argument tinkering, as in
   ``Env=Atari env.game=pong`` or ``Model=CNN model.depth=5`` (shown in
   `ways 1, 2, and 4 above <#way-1-purely-command-line>`__).
2. **Executable arguments** Executable code such as lists, tuples,
   dictionaries, and functions should be passed in quotes
   e.g. ``model.dims='[128, 64, 32]'``.
3. **Recipes** Note: we often use the “task” and “recipe” terms
   interchangeably. Both refer to the ``task=`` flag. `Ways 6 and 7
   above <#way-6-recipes>`__ show how to define a task/recipe.

Acceleration
------------

With ``accelerate=true``: \* Hard disk memory mapping. \* Adaptive RAM,
CUDA, and pinned-memory allocation & caching, with `customizable storage
distributions <>`__. \* Shared-RAM parallelism. \* Automatic 16-bit
mixed precision with ``mixed_precision=true``. \* Multi-GPU automatic
detection and parallel training with ``parallel=true``.

Fully supported across domains, including reinforcement learning and
generative modeling.

Tutorials
=========

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Custom architectures

.. raw:: html

   </h2>

.. raw:: html

   </summary>

.. code:: python

   # Run.py

   from torch import nn

   class Model(nn.Module):
      def __init__(self, in_features, out_features):
           super().__init__()

           self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

       def forward(self, x):
           return self.model(x)

**Run:**

.. code:: console

   ML Model=Run.Model Dataset=CIFAR10

Inferred shaping
----------------

UnifiedML automatically detects the shape signature of your model.

.. code:: diff

   # Run.py

   from torch import nn

   class Model(nn.Module):
   +   def __init__(self, in_features, out_features):
           super().__init__()

           self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

       def forward(self, x):
           return self.model(x)

Inferrable signature arguments include ``in_shape``, ``out_shape``,
``in_features``, ``out_features``, ``in_channels``, ``out_channels``,
``in_dim``, ``out_dim``.

Just include them as args to your model and UnifiedML will detect and
fill them in.

Thus, you can pass classes to command-line, not just objects.

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Custom datasets

.. raw:: html

   </h2>

.. raw:: html

   </summary>

Generally, a custom Dataset class may look like this:

.. code:: python

   # Run.py

   from torch.utils.data import Dataset

   class MyDataset(Dataset):
       def __init__(self, train=True):
           self.classes = ['dog', 'cat']
           ...

       def __getitem__(self, index):
           ...

           return obs, label

       def __len__(self):
           ...

For more info, see Pytorch’s tutorial on `map-style
Datasets <https://pytorch.org/docs/stable/data.html>`__.

**Run:**

.. code:: console

   ML Dataset=Run.MyDataset

**Classification**

Since the default task is ``task=classify``, the above script will learn
to classify ``MyDataset``.

If you define your own classify Dataset, include a ``.classes``
attribute listing the classes in your Dataset. Otherwise, UnifiedML will
automatically count unique classes, which may be different across
training and test sets.

**Test datasets**

You can include a ``train=`` boolean arg to your custom Dataset to
define different behaviors for training and testing, or use a different
custom test Dataset via ``TestDataset=``.

**Transforms & augmentations**

All passed-in Datasets will support the ``dataset.transform=`` argument.
``dataset.transform=`` is distinct from ``transform=`` and ``Aug=``, as
``transform=`` runs a transform on CPU at runtime and ``Aug=`` runs a
batch-vectorized augmentation on GPU at runtime, whereas
``dataset.transform=`` transforms/pre-compiles the dataset before
training begins. One-time operations like Resize are most efficient
here.

**Standardization & normalization**

Stats will automatically be computed for standardization and
normalization, and saved in the corresponding Memory ``card.yaml`` in
``World/ReplayBuffer``. Disable standardization with
``standardize=false``. This will trigger to use normalization instead.
Disable both with ``standardize=false norm=false``. You may learn more
about the differences at
`GeeksforGeeks <https://www.geeksforgeeks.org/normalization-vs-standardization/>`__.
By default, an agent loaded from a checkpoint will reuse its original
tabulated stats of the data that it was trained on even when evaluated
or further trained on a new dataset, to keep conditions consistent.

**Subsets**

Sub-classing is possible with the ``dataset.subset='[0, 5, 2]'``
keyword. In this example, only classes ``0``, ``5``, and ``2`` of the
given Dataset will be used for training and evaluation.

Training ImageNet
-----------------

Here’s how easy it is to start training on ImageNet-1k using the
built-in torchvision Dataset with a custom transform:

.. code:: console

   ML Dataset=ImageNet dataset.root='imagenet/' dataset.transform='transforms.Resize(64)'

``dataset.root=`` points to the location of the downloaded
`imagenet <https://www.image-net.org/download.php>`__ dataset.

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   How to write custom loss functions, backwards, optim, etc.

.. raw:: html

   </h2>

.. raw:: html

   </summary>

Let’s look at the ``Model`` `from Custom Architectures <#tutorials>`__:

.. code:: python

   # Run.py

   from torch.nn.functional import cross_entropy

   class Model_(Model):
       def learn(self, replay, logger):  # Add a learn(·) method to the Model from before
           batch = next(replay)

           y = self(batch.obs)

           loss = cross_entropy(y, batch.label)
           logger.log(loss=loss)

           return loss

**Run:**

.. code:: console

   ML Model=Run.Model_ Dataset=CIFAR10

We’ve now added a custom ``learn(·)`` method to our original ``Model``
that does basic cross-entropy.

For more sophisticated optimization schemes, we may optimize directly
within the ``learn(·)`` method
(e.g. ``loss.backward(); self.optim.step()``) and not return a loss.

```replay`` <World/Replay.py>`__ allows us to sample batches.
```logger`` <Logger.py>`__ allows us to keep track of metrics.

-  By the way, there’s no difference between ``Model=`` and ``Agent=``.
   The two are interchangeable. However, ``Model=`` in this example
   demonstrates a simplified version of the full capacity of Agents,
   which includes multi-task learning and *generalism*.
-  `We provide many Agent/Model examples across domains, including RL
   and generative modeling. <Agents>`__

.. _section-1:

Use ``Optim=`` or ``Scheduler=`` to define a custom optimizer or
scheduler:

.. code:: console

   ML Model=Run.Model_ Dataset=CIFAR10 Optim=Adam optim.lr=1e2 Scheduler=CosineAnnealingLR scheduler.T_max=1000

or one of the existing shorthands for the above-equivalent:

.. code:: console

   ML Model=Run.Model_ Dataset=CIFAR10 lr=1e2 lr_decay_epochs=1000

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Custom Environments

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Plotting, Logging, Stats, & Media

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Saving & Loading

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Multi-Task

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Multi-Modal

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Cheatsheet of built-in learning modes & features

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Cheatsheet: Built-in features of default Agent.learn

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Most of these features are implemented, but not yet documented. Please
sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

Examples
========

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   CIFAR10 in 10 seconds

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   ImageNet on 1 GPU

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Imagen: Text to image

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Stable Diffusion

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Humanoid from pixels

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   BittleBot: Real-time robotics with RL

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Image Segmentation

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Atari

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

.. raw:: html

   <h2>

   Text prediction

.. raw:: html

   </h2>

.. raw:: html

   </summary>

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__

.. raw:: html

   </details>

Reproducing works
=================

To be continued …

--------------

.. _section-2:

By `Sam Lerman <https://www.github.com/slerman12>`__.

`MIT license. <MIT_LICENSE>`__

`Please sponsor to support the development of this
work. <https://github.com/sponsors/AGI-init>`__
