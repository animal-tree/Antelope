Archetypal Psychology
========================

Concept-work, interim example:

   **Exploring and Mastering The Unknown in Deep Reinforcement Learning: The Hero’s Journey as Learning Algorithm**

   *Stories are an innate property of human psychology and
   consciousness. From primal days to the modern age, humans have been
   telling stories. Stories are so standard and commonplace in our lives
   that we usually take them for granted, as being naturally beautiful
   “things” in nature that just “are.” However, like with most art and
   music, it’s actually a profound mystery to contemplate what it is
   that makes and distinguishes a story as opposed to just random
   patterns of facts or memories, and what makes them so powerful and
   immersive for us at the deepest psychological levels. In this work,
   we revive and borrow a perspective of stories as a kind of “learning
   algorithm” for an individual’s psyche originating from fields as
   depth psychology (Jung) and Comparative Myth (Joseph Campbell). This
   resurrected perspective is studied with the more mathematically
   rigorous framework of AI, and given empirical support (under the new
   formulation) via surpassing state of the art results in deep
   reinforcement learning. We posit “archetypes” as agents both within
   and without a Hero’s psyche that, while cumbersome to fend off and
   deal with, ultimately facilitate the growth and learning of the
   hero-agent beyond what ordinary exploration methods (variational
   sampling / entropy, parameter noise) could afford. These archetypes
   are driven through a cyclic “journey” curriculum of oscillatingly
   high and low variationality. By exiting the familiar world and
   entering the realm of the unknown, encountering the various “faces”
   of differing inner psychological forces (“archetypes”), and
   ultimately returning — only to repeat the cycle again — our AI hero
   is able to “master both worlds” and play a number of standard
   benchmark games successfully.*

   |

   | Psychological journeys are rich and complex and highly-individual,
     tailored uniquely to the “hero” undergoing them, full of character
     faces, emotions, memories, and hardships that are unique to the
     individual and their life experience. The tapestry is as rich and
     complicated as the vast array of human beings who we might
     encounter, with each of their characters and personas lending a
     voice and a kind of aura to our processing of events. These
     life-specific relationships, “story elements” and chronologies
     can’t be mimicked or defined, but our processing of them can imbue
     them with some common structures. One of those structures, we
     posit, compresses “episodes” of information into learnable and
     memorable and communicable chunks which resemble what we instinctively identify
     as stories, and which highlight a certain range of exploration
     which leads to better action-taking in the future. This exploratory
     process can be conceived of as a narrative-style journey of 3
     higher-level acts: (1) leaving the familiar world, (2) entering the
     unknown realm, (3) and returning, a cycle of less exploration to
     more exploration to less exploration. At the end, the hero is
     accredited the title “master of two worlds,” familiar and unknown.

   By oscillating exploration cyclically, an agent can exploit current
   knowledge, discover new knowledge, then exploit the new knowledge,
   thereby collecting a wider range of positive, negative, and new
   experience from which to learn, not over-fitting to any one strategy.

   Commonly, exploration rates are decayed linearly or according to a
   monotonic schedule. While this leads to more deterministic policies
   over time as the agent improves, it also can result in

   convergence to local optima. In contrast, cyclic exploration leads to
   perpetual re-learning and discovery capacity.

   “Archetypes” appear in stories as helpful or villainous forces that
   steer the hero along in their psychological journey. They can be
   represented externally as characters and relationships that the hero
   encounters: allies, enemies; or internally as faces of the hero
   protagonist that were perhaps not previously known to him. Defeating
   the villain (“adversary”) is often regarded metaphorically as
   “integrating the shadow” for example. A “trickster” archetype can
   emerge to set the world in unbalance, only to catalyze the hero’s
   ultimate growth. The hero himself may yearn for adventure, called to
   it by external forces, or metaphorically by the hero’s own
   exploratory spirit.

   We define three “archetypes” for the sake of learning: 1. The hero
   himself, 1. An explorer, 2. An adversary / trickster. Thereby, we
   disentangle the single agent into three agents which we call
   “archetypes”. For evaluation, we employ only “the hero” with an
   exploration rate of 0, but for training, we obtain experience from
   all 3 under a cyclic exploration rate that we refer to as our “Hero’s
   Journey.”

   Each archetype is distinguished by a different objective function.
   Furthermore, each is trained equally and indiscriminately from the
   collective memory. They are deployed during training equally for
   experience collection which fills up a shared replay memory and their
   psychological tendencies (that is, neural network parameters) are
   updated from shared batches drawn from said replay. Their function is
   to promote a different exploration style each and maneuver the hero
   to learning from avenues of exploration previously unknown and not
   simply discovered through mere random sampling.

   We hope that this new perspective of “computational depth psychology”
   inspires future work to define psychological or artistic-cognitive
   processes in more rigorous, math-based terms, if not just simplest
   analog formulations to start, and reproducible methodologies for
   testing their inductive likelihood as models of the brain and the
   human psyche. Loosely bridging two highly distinct schools, or
   “worlds” perhaps, familiar and unknown from the perspective of each.
