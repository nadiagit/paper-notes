# paper-notes-and-conferences
## EEML2020
**David Silver - Challenges in Reinforcement Learning**  
A system that understands for itself how to solve a problem.  
Long-term goals that we’ll like AI to do is the RL problem.  
Hypothesis: The RL problem is actually the problem that captures everything we need for intelligence?  
So, by studying that problem and finding ways to solve that one problem we may be able to go an enormous distance towards general intelligence.   

**Challenge**  
An agent will be bombarded with a stream of sensations that might come from vision, or audio, or robotic sensors or whatever that data is. The agent will inhabit this source of observations coming in and actions going out. We have to be able to solve those streams of information. If we pursue general intelligence in terms of human intelligence, we’re able to do lifelong learning, we have long time scales where the agent is borned and starts to being bombarded with the sensations the first moment of its life until 17-18 years and it has to keep building and constructing knowledge and understanding how to parse this reach stream of experiences better and better.  
That is the challenge, we made progress to know how to deal with problems at shorter time scales. Long lifetimes feels like a big next step to take.
  
RL and applicability in industry  
Adaptive control with models built in by people that explains how the dynamic of a system works. Built more applications where the system learns for itself through experience: how their world works, what the model is, what are the dynamics.  
Healthcare, online interactions.



**Irina Higgins**  
  How to address the limitations of deep reinforcement learning algorithms? It seems that unsupervised learning might be useful. An algorithm that can learn about this world the same way babies do. Explore how the world works, so when supervision does come, it can do so much faster and robustly than deep learning systems.  
If you don’t have ground truth supervision what do you do? One approach: look for inspiration in neuroscience, physics, linguistics.  
Disentangled representation learning: without any supervision, by just observing the data, figure out what transformations give rise to that data.   
Article: https://arxiv.org/pdf/1901.11390.pdf  
* A 3D scene with objects inside  
* Train a autoencoder architecture which takes the input and tried to reconstruct it with particular regularization inside   
  * They arrived to single units inside this network that control interetable aspects of the environment. If we modify the values of one of these units and then try to reconstruct, we see that iit only affects for example the color of one of the objects, or its size or the composition.  
* Inspired by physics and neuroscience  
* Utility:  
  * Data efficiency  
  * Generalisation under covariate shift  
  * Fairness  
  * Abstract reasoning  
  * Transfer



## Using GANs and Genetic Algorithms to generate video game levels.  
* Article: https://arxiv.org/pdf/2004.01703.pdf  
* GANs have shown great potential in recent years to generate convincing organic images, sounds and even videos  
* Tasks with complex functional constraints are hard for them, such as generating video game levels, where much like computer programs, looking convincing is not enough if the level is not beatable. To address this, Schurm, Volz, et al. bring us two (1, 2) very interesting papers that merge newfangled GANs with traditional evolutionary methods to generate levels that are both good-looking and actually playable!  
* The key innovation here is using a technique called Latent Variable Evolution (LVE), in which a GAN is trained to generate levels (for Mario, Zelda, Sokoban, etc) as if they were images, but the latent space used is then searched using evolutionary methods, according to some arbitrary fitness function.  
* The authors are then free to define whatever function they want, such as how easily an agent can beat the level or whether some human-in-the-loop decides if they like it or not. This frees the model from having to learn such difficult constraints, constraints which not only are non-differentiable but may just be too hard even for modern GANs.  


## Generative adversarial networks  
* Article: https://arxiv.org/abs/1406.2661  
* Simultaneously train two models  
  * A generative model G that captures the data distribution,  
  * A discriminative model D that estimates the probability that a sample came from the training data rather than G.  
* The training for G is to maximize the probability of D making a mistake.  
* D and G play a two-player minimax game.  
  * k steps of optimizing D for one step of optimizing G,  
  * Early in learning, D can reject samples with high confidence, because they are clearly different from the training data  
    * G trained to maximize log D (G(z)).  
* Theoretic result: the optimal distribution for D is:  
     <img src="https://latex.codecogs.com/svg.latex?D_{G}^{*}&space;=&space;\frac{p_{data}(x)}{p_{data}(x)&space;&plus;&space;p_{g}(x)}" title="D_{G}^{*} = \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)}" />
 
## Playing Atari with Deep Reinforcement Learning  
* Article: https://arxiv.org/pdf/1312.5602v1.pdf  
* Learning control policies: CNN with  
  * input: raw pixels  
  * output: a value function estimating future rewards

DL methods | RL  
---------- | --  
Require large amounts of hand-labelled training data; | Must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed;  
Direct association between inputs and targets found in supervised learning; | Delay between actions and resulting rewards which can be thousands of timesteps long;  
Assumes data samples to be independent; | Typically encounters sequences of highly correlated states;  
Assume a fixed underlying distribution. | The data distribution changes as the algorithm learns new behaviours.

**Goal** : create a single neural network agent that is able to successfully learn to play as many of the games as possible.  
  * The network has not been provided with any game-specific information, it learned from nothing but the video input, the reward and terminal signals, and the set of possible actions.  
* How? Applying **Q learning** to Atari, approximate the Q value using a conv net.   
* Agent interacts with environment ε (Atari emulator) =>  
  * Sequence of actions
  * Observations
  * Rewards
    * Each time step, agent selects an action <img src="https://latex.codecogs.com/svg.latex?a_{t}" title="a_{t}" />
      * Passed to the emulator -> modifies internal state and score.
      * Agent 
        * Observes an image <img src="https://latex.codecogs.com/svg.latex?x_{t}&space;\in&space;\mathbb{R}^{d}" title="x_{t} \in \mathbb{R}^{d}" /> the emulator (vector of raw pixel values representing the current screen).
        * Receives a reward <img src="https://latex.codecogs.com/svg.latex?r_{t}" title="r_{t}" /> = change in game score.
* MDP which each sequence is a distinct state.
* **Agent goal**: interact with the emulator by selecting actions in a way that maximises future rewards
  * The maximum expected return achievable -> optimal action-value function
<img src="https://latex.codecogs.com/svg.latex?Q^{*}&space;(s,&space;a)&space;=&space;max&space;\pi&space;E&space;[\mathbb{R}_{t}&space;|&space;s_{t}&space;=&space;s,&space;a,&space;\pi&space;]" title="Q^{*} (s, a) = max \pi E [\mathbb{R}_{t} | s_{t} = s, a, \pi ]" />   

  * <img src="https://latex.codecogs.com/svg.latex?Q^{*}" title="Q^{*}" />obeys an important identity known as the Bellman equation.    
    * Impractical in practice because the action-value function is estimated separately for each sequence, without any generalization.  
    * Common to use a function approximator to estimate action-value function
        * Linear function approximator
        * Non-linear function approximator
          * Neural network with weights as a **Q-network**: <img src="https://latex.codecogs.com/svg.latex?Q&space;(s,&space;a;&space;\theta&space;)&space;\approx&space;Q^{*}&space;(s,&space;a)" title="Q (s, a; \theta ) \approx Q^{*} (s, a)" />
