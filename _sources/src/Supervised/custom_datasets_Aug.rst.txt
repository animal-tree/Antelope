Custom Datasets, Aug
====================

Generally, a custom Dataset class may look like this:

.. code:: python

   from torch.utils.data import Dataset

   from antelope import ml

   class MyDataset(Dataset):
       def __init__(self, train=True):
           self.classes = ['dog', 'cat']
           ...

       def __getitem__(self, index):
           ...

           return obs, label

       def __len__(self):
           ...

   ml(dataset=MyDataset)

For more info, see Pytorch’s tutorial on `map-style
Datasets <https://pytorch.org/docs/stable/data.html>`__.

Any Pytorch Dataset will do, with one caveat: the ``__getitem__`` should output either an ``(obs, label)`` pair OR a dictionary of "datums" e.g. ``{'obs': obs, 'label': label}``.

**Classification**

Since the default task is ``task=classify``, the above script will learn
to classify ``MyDataset``.

If you define your own classify Dataset, include a ``.classes``
attribute listing the classes in your Dataset. Otherwise, The Antelope will
automatically count unique classes, which may be different across
training and test sets.

**Test datasets**

You can include a ``train=`` boolean arg to your custom Dataset to
define different behaviors for training and testing, or use a different
custom test Dataset via ``test_dataset=``.

**Transforms & augmentations**

Pre-compile:

.. code:: python

   from torchvision.transforms import Resize

   ml(dataset={'_target_': MyDataset, 'transform': Resize([302,170])})

Transform on CPU at runtime:

.. code:: python

   ml(dataset=MyDataset, transform=Resize([302,170]))

Transform on GPU at runtime:

.. code:: python

   from antelope.Agents.Blocks.Augmentations import RandomShiftsAug

   ml(dataset=MyDataset, aug=RandomShiftsAug)

All passed-in Datasets will support the ``dataset.transform=`` argument.
``dataset.transform=`` is distinct from ``transform=`` and ``aug=``, as
``transform=`` runs a transform on CPU at runtime and ``aug=`` runs a
batch-vectorized augmentation on GPU at runtime, whereas
``dataset.transform=`` transforms/pre-compiles the dataset before
training begins. One-time operations like Resize are most efficient
here.

There are also two additional kinds of transform/augmentation for nuanced cases. ``env.transform=`` can transform an online stream from a rollout at runtime and ``dataset.aug=`` can pre-compile a dataset with a batch-vectorized augmentation on GPU. Pretty much every transform/aug need is met in whatever data processing pipeline you're trying to implement.

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

.. code:: python

   ml(dataset={'_target_': MyDataset, 'subset': [0, 5, 2]})

^^^^^^^^^^^^^^^^^^^^
Built-In Datasets
^^^^^^^^^^^^^^^^^^^^

All `TorchVision datasets <https://pytorch.org/vision/main/datasets.html>`__ are supported by default and can be passed in by name (e.g. ``dataset=MNIST``) as well as ``TinyImageNet``, which is provided as an example custom dataset.

----

For an `iterative-style Dataset <https://pytorch.org/docs/stable/data.html>`__, use an `Environment <../Reinforcement-Learning/custom_env_&_custom_reward.html>`__.