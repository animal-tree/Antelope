Mere basics
=======================

^^^^^^^^^^^^^^^^^^^
Custom architecture
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch import nn
    from torch.nn.functional import cross_entropy

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

Besides ``in_features``, ``out_features`` you can pass an array of inferrable shape parameters, including ``in_shape``, ``out_shape``,
``in_features``, ``out_features``, ``in_channels``, ``out_channels``,
``in_dim``, ``out_dim``.

Just include them as args to your model and The Antelope will detect and
fill them in.

^^^^^^^^^^^^^^^^^^^
Custom learning
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Model(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()

            self.MLP = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features, 32),
                                     nn.Linear(32, out_features))

        def forward(self, x):
            return self.MLP(x)

        def learn(self, replay):
            batch = next(replay)
            y = self(batch.obs)
            loss = cross_entropy(y, batch.label)
            return loss

    ml(model=Model, dataset='CIFAR10')

^^^^^^^^^^^^^^^^^^^
Advanced learning
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Model(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()

            self.MLP = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features, 32),
                                     nn.Linear(32, out_features))

            self.loss = torch.nn.CrossEntropyLoss()
            self.optim = torch.optim.Adam(self.MLP.parameters(), lr=1e-4)

        def forward(self, x):
            return self.MLP(x)

        def learn(self, replay, log):
            self.optim.zero_grad()
            batch = next(replay)
            y = self(batch.obs)
            loss = cross_entropy(y, batch.label)
            loss.backward()
            self.optim.step()
            log.loss = loss

    ml(model=Model, dataset='CIFAR10')

^^^^^^^^^^^^^^^^^^^
Logging
^^^^^^^^^^^^^^^^^^^

.. code:: diff

    class Model(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()

            self.MLP = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features, 32),
                                     nn.Linear(32, out_features))

        def forward(self, x):
            return self.MLP(x)

        def learn(self, replay, log):
            batch = next(replay)
            y = self(batch.obs)
            loss = cross_entropy(y, batch.label)

    +       log.log_five = 5
    +       log.my_loss = loss
    +       log.UnifiedML = 42

            return loss


Matrices can be automatically averaged by logger. Try:

- ``log.array = torch.ones([2])  # Arrays will be cumulatively averaged by logger``

^^^^^^^^^^^^^^^^^^^
Paths
^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Run.py

   from torch import nn

   model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))  # Two-layer neural-net

**Run:**

.. code:: console

   ml model=Run.model dataset=CIFAR10

This demonstrates **dot notation** (``Run.model``) for pointing
to an antelope object. Equivalently, it’s possible to use **regular directory
paths**:

.. code:: console

   ml model=./Run.py.model dataset=CIFAR10

Wherever you run ``ml``, it’ll search from the current directory for any
specified paths.

If you have a custom dataset for example:

.. code:: console

   ml model=Run.model dataset=path.to.MyCustomDataset

Besides the current directory, the search path includes Antelope's root directory.

Built-in paths can be accessed directly. For example, the built-in ``CNN`` architecture can be referenced just by ``model=CNN`` without needing to specify the full path (as in, ``model=Agents.Blocks.Architectures.Vision.CNN``).

^^^^^^^^^^^^^^^^^^^
Synonyms & a billion universes
^^^^^^^^^^^^^^^^^^^

Here’s how to write the same program in 7 different ways.

.. raw:: html

   </h3>

.. raw:: html

   </summary>

Train a simple 5-layer CNN to play Atari Pong:

Way 1. Purely command-line
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   ml task=RL env=Atari env.game=pong model=CNN model.depth=5

Way 2. Command-line code
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   ml task=RL env='Atari(game="pong")' model='CNN(depth=5)'

Way 3. Command-line
~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Run.py

   from antelope import ml

   ml()

**Run:**

.. code:: console

   python Run.py task=RL env=Atari env.game=pong model=CNN model.depth=5

Way 4. Inferred Code
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Run.py

   from antelope import ml

   ml('env.game=pong', 'model.depth=5', task='RL', env='Atari', model='CNN')

**Run:**

.. code:: console

   python Run.py

Way 5. Purely Code
~~~~~~~~~~~~~~~~~~

.. code:: python

   # Run.py

   from antelope import ml
   from antelope.Blocks.Architectures import CNN
   from antelope.World.Environments import Atari

   ml(task='RL', env=Atari(game='pong'), model=CNN(depth=5))

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

   ml task=recipe

The ``imports:`` syntax allows importing multiple tasks/recipes from
different sources, with the last item in the list having the highest
priority when arguments conflict. In this case, writing ``task: RL`` would be equivalent.

The capitalized args correspond to the ``_target_:`` sub-arg of their lowercase counterparts. They're just a cleaner shorthand for the usual ``_target_:`` initialization syntax (see below: Way 7).

Custom task ``.yaml`` files will be searched for in the root directory
``./``, a ``Hyperparams/`` directory if one exists, and a
``Hyperparams/task`` directory if one exists.

Way 7. All of the above
~~~~~~~~~~~~~~~~~~~~~~~

The order of hyperparam priority is ``command-line > code > recipe``.

Here’s a combined example:

.. code:: yaml

   # recipe.yaml

   task: RL
   model:
     _target_: CNN
     depth: 5

.. code:: python

   # Run.py

   from antelope import ml
   from antelope.World.Environments.Atari import Atari

   ml(env=Atari)

**Run:**

.. code:: console

   python Run.py task=recipe env.game=pong

.. raw:: html

   </details>

----

**Keep reading for how to extend this to everything you need, from ImageNet to real-time robotics, all at the palms of your fingertips.** |:palm_tree:|