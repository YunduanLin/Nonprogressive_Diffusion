# Code for the paper -- Nonprogressive Diffusion on Social Networks: Approximation and Applications

Here the authors provide the code used in the paper, [Nonprogressive Diffusion on Social Networks: Approximation and Applications](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4232670). In this work, the authors use the open-source social network data on [Network Repository](https://networkrepository.com/networks.php) and synthetic random graph data. In the paper, in order to robustly test the performance of their approximation algorithm, the authors test over a large set of networks with different parameter configurations and with multiple repititions. In this repository, we include code and data for one synthetic random graph and one real-world for illustration. All other instances included in the paper can be replicated by changing the specific parameter. You could contact the authors if you have any questions.

## Files

---

### code

network.py

- This file includes a class that is used to represent a specific social network. We can either store a given network structure from the file, or generate random graph networks with given parameter configurations.

diffusion_simulation.py

- This file includes a class that is used to establish the network diffusion environment. It includes both the agent-based simulation method and the fixed-point approximation scheme of obtaining agents' long-term adoption probabilities.

IM.py

- This file includes the algorithms for solving the influence maximization problem, which is an application introduced in the paper.

pricing_gradient.py

- This file includes the algorithms for solving the pricing problem on a social network, which is another application introduced in the paper.

test.ipynb

- This is an example test instance. 

### instances



