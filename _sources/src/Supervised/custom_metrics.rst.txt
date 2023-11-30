custom metrics
==============

The difference between logs and metrics is that metrics are computed at inference or environment rollouts. Logs are computed during training when the learn method is called, as seen in `mere basics <../mere_basics.html>`__.

* If you're using the default Agent ``learn``, not your own custom one, then you can add a ``agent.log=true`` to enable printing extra logs from that.

See `saving & loading <../App/saving_&_loading.html>`__ for info on how to save/dump entire features/representations/embeddings computed during the last evaluation.

^^^^^^^^^^^^^^^^^^^
Accuracy
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Accuracy:
        # An experience is a set of batch data that follows an action
        def add(self, exp):
            return exp.label == exp.action  # Gets appended to an epoch list

        # At the end of an epoch, a metric is tabulated
        def tabulate(self, epoch):
            return epoch  # Lists/arrays get concatenated and mean-averaged by default

    ml(metric={'accuracy': Accuracy})

Accuracy is already computed for the classify task by default, which is the default.

The default dataset is MNIST.

^^^^^^^^^^^^^^^^^^^
MSE
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class MSE:
        def add(self, exp):
            return (exp.label - exp.action) ** 2  # Gets appended to an epoch list

        def tabulate(self, epoch):
            return epoch  # Lists/arrays get concatenated and mean-averaged by default

    ml(metric={'mse': MSE})

MSE is already computed for the regression task (``task=regression``).

^^^^^^^^^^^^^^^^^^^
Reward
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Reward:
        def add(self, exp):
            if 'reward' in exp:
                return exp.reward.mean() if hasattr(exp.reward, 'mean') else exp.reward

        def tabulate(self, episode):  # At the end of an episode, a metric is tabulated
            if episode:
                return sum(episode)

    ml(metric={'reward': Reward})

Reward is already computed whenever an environment returns a reward, such as in reinforcement learning.

See `Custom Env & custom reward <../Reinforcement-Learning/custom_env_&_custom_reward.rst>`__ for how this metric can be used to override the reward function.

^^^^^^^^^^^^^^^^^^^
Precision
^^^^^^^^^^^^^^^^^^^

(TODO these aren't mathematically correct at the moment)

.. code-block:: python

    class Precision:
        def add(self, exp):
            classes = np.unique(exp.action)

            true_positives = {c: (exp.action == exp.label) & (exp.action == c) for c in classes}
            total = {c: sum(exp.action == c) for c in classes}

            return true_positives, total

        def tabulate(self, epoch):
            # Micro-average precision
            return sum([true_positives[key] for true_positives, total in epoch for key in true_positives]) \
                / sum([total[key] for true_positives, total in epoch for key in true_positives])

    ml(metric={'precision': Precision})

^^^^^^^^^^^^^^^^^^^
Recall
^^^^^^^^^^^^^^^^^^^

(TODO these aren't mathematically correct at the moment)

.. code-block:: python

    class Recall:
        def add(self, exp):
            classes = np.unique(exp.label)

            true_positives = {c: (exp.action == exp.label) & (exp.action == c) for c in classes}
            total = {c: sum(exp.label == c) for c in classes}

            return true_positives, total

        def tabulate(self, epoch):
            # Micro-average precision
            return sum([true_positives[key] for true_positives, total in epoch for key in true_positives]) \
                / sum([total[key] for true_positives, total in epoch for key in true_positives])

    ml(metric={'recall': Recall})


^^^^^^^^^^^^^^^^^^^
F1-score
^^^^^^^^^^^^^^^^^^^

(TODO these aren't mathematically correct at the moment)

.. code-block:: python

    ml(metric={'precision': Precision, 'recall': Recall,
               'f1': '2 * precision * recall / (precision + recall)'})

Metrics can be evaluated from strings based on other metrics.

^^^^^^^^^^^^^^^^^^^
Via command-line
^^^^^^^^^^^^^^^^^^^

Suppose the above metrics are defined in a ``Run.py`` file.

Either

.. code-block:: python

    # Run.py
    ...

    ml()  # Add a call to ml() in Run.py

.. code-block:: console

    python Run.py metric.precision=Run.Precision metric.recall=Run.Recall metric.f1=2*precision*recall/(precision+recall)

or

.. code-block:: console

    ml metric.precision=Run.Precision metric.recall=Run.Recall metric.f1=2*precision*recall/(precision+recall)

.. note:: The ``ml`` console-command is automatically installed with antelope-brethren-star. It accepts everything the code-based and `yaml <../App/defining_tasks.html>`__ interfaces do.

^^^^^^^^^^^^^^^^^^^
Via Yaml
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # My_Recipe.yaml
    task: classify
    metric:
        precision: Run.Precision
        recall: Run.Recall
        f1: '2 * precision * recall / (precision + recall)'

.. code-block:: console

    ml task=My_Recipe