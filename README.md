# Conservative_PINNs

We propose a conservative physics-informed neural network (cPINN) on discrete domains for nonlinear conservation laws. 
Here, the term discrete domain represents the discrete sub-domains obtained after division of the computational domain, where PINN is
applied and the conservation property of cPINN is obtained by enforcing the flux continuity in the strong form along the sub-domain interfaces.
In case of hyperbolic conservation laws, the convective flux contributes at the interfaces, whereas in case of viscous conservation laws,
both convective and diffusive fluxes contribute. Apart from the flux continuity condition, an average solution (given by two different neural networks)
is also enforced at the common interface between two sub-domains. One can also employ a deep neural network in the domain, where the solution may
have complex structure, whereas a shallow neural network can be used in the sub-domains with relatively simple and smooth solutions. 
Another advantage of the proposed method is the additional freedom it gives in terms of the choice of optimization algorithm and the 
various training parameters like residual points, activation function, width and depth of the network etc. Various forms of errors involved
in cPINN such as optimization, generalization and approximation errors and their sources are discussed briefly. In cPINN, locally adaptive
activation functions are used, hence training the model faster compared to its fixed counterparts. Both, forward and inverse problems are 
solved using the proposed method. Various test cases ranging from scalar nonlinear conservation laws like Burgers, Korteweg–de Vries (KdV)
equations to systems of conservation laws, like compressible Euler equations are solved. The lid-driven cavity test case governed by incompressible
Navier–Stokes equation is also solved and the results are compared against a benchmark solution. The proposed method enjoys the property of domain 
decomposition with separate neural networks in each sub-domain,
and it efficiently lends itself to parallelized computation, where each sub-domain can be assigned to a different computational node.
