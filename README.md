# self-adaptive-IM
Matlab code simulating self-adaptive Ising machines (SAIM) for constrained combinatorial optimization. SAIMs are based on a Lagrange relaxation of constraint that iteratively shapes the energy landscape and brings the ground states to satisfiable regions.
For more information, please see the arxiv paper:
https://arxiv.org/pdf/2501.04971

The Matlab file SAIM_QKP.m executes a probabilistic-bit (p-bit) SAIM for quadratic knapsack problems. SAIM_MKP.m addresses multidimensional knapsack problems.

The multidimensional Knapsack problem (MKP) instances are from the paper by P.C. Chu and J.E. Beasley, "A Genetic Algorithm for the Multidimensional Knapsack Problem", Journal of Heuristics, 1998. They are available at:
https://home.himolde.no/hvattum/benchmarks/BIP/index.html

The quadratic knapsack problem (QKP) instances are from the paper by A. Billionnet and E. Soutif, "An exact method based on Lagrangian decomposition for the 0–1 quadratic knapsack problem', European Journal of Operational Research, 2004. They are available at:
https://cedric.cnam.fr/~soutif/QKP/
