Custom Learning
==============

.. code-block:: python

    from antelope import ml

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


| ``replay`` allows us to sample batches.

-  By the way, there’s no difference between ``model=`` and ``agent=``.
   The two are interchangeable.
-  `We provide many Agent/Model examples across domains, including RL
   and generative modeling. <Agents>`__
- Besides the shaping parameters, many other initialization args can be inferred, included, or excluded, ``obs_spec`` and ``action_spec`` for example.

Use ``optim=`` or ``scheduler=`` to define a custom optimizer or
scheduler:

.. code:: python

   from torch.optim import AdamW
   from torch.optim.lr_scheduler import CosineAnnealingLR

   ml(model=Model, dataset='CIFAR10', optim={'_target_': AdamW, 'lr': 1e2}, scheduler={'_target_': CosineAnnealingLR, 'T_max': 1000})

or one of the existing shorthands for the above-equivalent:

.. code:: python

   ml(model=Model, dataset='CIFAR10', lr=1e2, lr_decay_epochs=1000)

Here is a console example, not using the shorthands:

.. code:: console

   ml optim=SGD optim.lr=1e-2 scheduler=CosineAnnealingLR scheduler.T_max=1000

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

No need to return a loss in the ``learn`` method.

| ``log.`` allows us to keep track of logging metrics.

^^^^^^^^^^^^^^^^^^^
Ultra-advanced learning
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from math import prod

    class Model(nn.Module):
        def __init__(self, spec):
            super().__init__()

            in_features = prod(spec.obs.shape)
            out_features = prod(spec.action.shape)

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

``spec`` can provide more thorough information about an environment's specs and its different "datums", including statistics (``spec.obs.mean``, ``spec.obs.stddev``, ``spec.obs.low``, ``spec.obs.high``) and action space (``spec.action.discrete``, ``spec.action.low``, ``spec.action.high``).

See `AC2Agent <Agents>`__, a single Agent/Model example that does everything from classification to regression to RL to generative modeling to action space conversion. This barebones backbone is The Antelope's default agent.