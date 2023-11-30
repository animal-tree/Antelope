a GI
==================


    a GI can refer to a gastro-intestinal disorder or "general intelligence". "General intelligence" means something different but similar to artificial general intelligence. Whereas artificial general intelligence deals with computers and machines, and learning algorithms as constructed from those theological points of view, "general intelligence" admits a different theology that is more societal, individual, and personal. It says, we together must be intelligent.

    However, to understand our own intelligence and perhaps the intelligence of the cosmos, if such a thing exists, it helps to (1) study nature and our relationship with it, (2) study nature and its biological and physical properties, and (3) deduce algorithmically our own constructed representations and models that could describe, emulate, or even outperform.

    Should we build AGI? Well, to me the question is like, "should we build fusion and fission technologies?"

    Perhaps solar would suffice, and perhaps the existing toolbox is more than enough to provide for the wants and needs of every human.

    However, the understanding that this technology bridges is between representations we can understand through our logic and ones that are extraordinary and deeply profound. In that purpose, general intelligence is served as we better understand Mother Nature and Her extreme creation (or destruction) potential. Through AGI, we can better understand the logics of intelligence (generally) that govern our own minds and psyches, and perhaps the universe. We can then use that knowledge, to, for example, cure gastro-intestinal disorders.

..
    ?" Honestly, it seems overkill to our actual needs. Most energy use is industrial and most industry is pretty wasteful. But if we need to split the atom to make Snickers bars, so be it.

    Perhaps solar would suffice, and perhaps the existing toolbox is more than enough to provide for the wants and needs of every human.

    But issues remain. From climate change to ravaging diseases. It's unclear but suspected that AGI could go a significant way to addressing problems we may not have alternative solutions for, maybe.

    for (maybe).

    Maybe because there are still many things we haven't tried.

    We haven't tried:

    (a) A societal and economic system that provides for everyone, successfully.

    (b) The substitution of humanity's huge disconnect from nature with an integration of nature into our infrastructure, ritual, culture, and daily lives as per human heart.

    (c) Emulating more evolutionary conditions but with the advantage of our developed irrigation and pastoral-agricultural systems.

    (d) Cleaning up. (Pollution) Then breathing the air with our industrial developments sustainably in hand not working against us.

    (e) Opening our hearts in greater ways to strangers and developing a more mature standard culture where perception of others, understanding, and forgiveness are all innate principles, and we each hold each other as sacred as we would the animals and trees we love, measuring our success not by capitalistic wealth but by the wealth (in all its meanings) of our lowest individual.

    Which directions are as necessary or worth trying as "AGI" is unclear.

    However, the understanding that this technology bridges is between representations we can understand through our logic and ones that are extraordinary and deeply profound. In that purpose, general intelligence is served as we better understand Mother Nature and Her extreme creation (or destruction) potential. Through AGI, we can better understand the logics of intelligence (generally) that govern our own minds and psyches, and perhaps the universe. We can then use that knowledge, to, for example, cure gastro-intestinal disorders.
..

..
    ##########
    ideas
    ##########
..

..
    Multi-block framework for all Ai, including Time, Multi-Model, and Multi-Env
    Stable Diffusion, Imagen
    Eventually distributed, 4-bit quantized, JIT'ed and so forth
    GATO
    ~JEPA

    pretty much

    Our built-in agent can probably already do much of what you might want to. If you'd prefer not to rebuild the wheel, we provide a "Blocks" API for customizing the built-in agent's architectures and more.

    The Blocks include, generally: Aug, Encoder, Actor and Critic, Creator.

    Purposes:

    Aug: GPU-based augmentation (faster than CPU-based Transform).
    Encoder: Just Eyes and Pool.
    Actor: trunk/Pi_trunk and Pi_head/Generator. This guy outputs the action.
    Critic: trunk/Q_trunk and Q_head/Predictor/Discriminator. This one critiques the Actor, and the Actor maximizes accordingly. Useful for reward-based optimization and GANs.
    Creator: outputs "policies". Sampling distributions based on the Actor's action. Normals for continuous spaces and Categoricals for discrete spaces. Useful for exploration in RL.

    Examples:

    `ml aug=IntensityAug`
    `ml eyes=ResNet18 eyes.optim=SGD eyes.optim.lr=1e-2`
..