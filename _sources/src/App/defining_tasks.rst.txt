defining tasks
===============

^^^^^^^^^^^^^^^^^^^
describing magical potions
^^^^^^^^^^^^^^^^^^^

Did you know antelope-brethren-star has a command-line syntax? Take a look

.. code-block:: console

    ml model='nn.Sequential(nn.Flatten(1),nn.Linear(784,10))'

Therefore, we allow you to pass in paths, of your choosing

.. code-block:: console

    ml model=path.to.my_model.MyModel

^^^^^^^^^^^^^^^^^^^
By yaml
^^^^^^^^^^^^^^^^^^^

Another way to describe magical potions is by .yaml files

.. code-block:: yaml

    # path/to/my_magic_alchemy.yaml
    task: classify
    model: my_model.MyModel

.. code-block:: console

    ml task=path/to/my_magic_alchemy

Hyperparams that can be specified include ``model``, ``dataset``, and so much more. You have no idea

In short, tasks are ``.yaml`` files that override `Antelope's default hyperparams/args <Hyperparams/>`__.

^^^^^^^^^^^^^^^^^^^
built-in
^^^^^^^^^^^^^^^^^^^

The built-in defined tasks are

* `classify.yaml <github.com>`_ (default)
* `regression.yaml <github.com>`_
* `RL.yaml <github.com>`_
* `generate.yaml <github.com>`_

These can be called simply like

.. code-block:: console

    ml task=RL

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

