Custom Env & custom reward
==================

The run script does evaluation rollouts from the start to the end of the episode everytime it needs to evaluate. The ```evaluation_episodes``` flag can control how many evaluation rollouts it does, averaging them. In addition to these inference rollouts, the run script also does a 1-step *exploration* rollout at each step. In the *offline* training mode (```offline=true```), the exploration rollout is ignored by the environment. That is, the agent doesn't require exploration to obtain data, but rather pre-loads it from an existing data source (replay). In the case of ```stream=true```, no data is pre-loaded, even during offline. Instead of fetching data from the replay, data is "streamed" directly from the environment to the agent, transiently passing through the replay only in formality to allow the ```next(replay)``` syntax for sampling it. This is conventionally called ```on-policy``` in reinforcement learning. Data is obtained from the environment, even in the offline setting under the stream protocol. The distinguishing factor of the offline setting, where data is sampled rather than obtained through agent action, under the stream protocol, is that the agent is not called in order to step the exploration rollout in the offline setting, since data can be obtained independent of the agent's action. The exploration rollout is stepped without argument. In the exploration rollout of the offline-stream mode, a None-argument is presumed to be supported by the environment's step function for offline-stream to work. Meanwhile, ```stream``` is independent to the inference rollout protocol. In the ```online``` setting, the data from the environment's exploration steps is stored, and sampled from the ```replay``` as if it were part of the dataset. The syntax for sampling data is consistently always ```batch = next(replay)``` or ```batch = replay.sample()```. During these rollout modes, no gradient computations occur, training-only randomizations like dropout are temporarily disabled, and EMA may be toggled with the ```ema=true``` flag. Thus, rollouts have two distinct modes, one for evaluation/inference and another for exploration to collect data. Rollouts — both of these — are distinct from agent learning where gradient computations and optimizations happen. Ah, but there is one more pigment to this mosaic and that is generative mode with the ```generate=true``` flag, not described here. As well as many others here, from controlling the truncation of episode storage to, most impressively, the metrics. Quick custom defining of evaluative and reinforcement metrics. Perhaps more. Action repeats, frame stack...

.. code-block:: python
    from vidgear.gears import CamGear


    class YouTube:
    """
    Live-streaming environment
    """

    def __init__(self, url, train=True, steps=1000):
        url = url.split('?feature')[0]

        self.video = CamGear(source=url, stream_mode=True, logging=True).start()

        self.train = train
        self.steps = steps  # Controls evaluation episode length
        self.episode_step = 0

    def step(self, action=None):
        return self.reset()

    def reset(self):
        self.episode_step += 1
        return {'obs': as_tensor(self.video.read()).permute(2, 0, 1).unsqueeze(0),
                'done': not self.train and not self.episode_step % self.steps}