# I-AMOD-ride-pooling
This project proposes an approach to synthesize the works by [Paparella et al.](https://ieeexplore.ieee.org/document/10605118) on ride-pooling and [Wollenstein-Betech et al.](https://ieeexplore.ieee.org/document/9541261) on Intermodal Autonomous Mobility on Demand (I-AMoD). Both approaches will be combined to form a model that optimizes traffic flow for NYC, with data taken from [source]. 

# Multi-commodity network flow problem
Both the ride-pooling and I-AMoD problems can be modeled using multi-commodity network problems. The network can be defined as $\mathcal{G}$, consisting of a road layer and $\mathcal{L}$ additional layers representing other modes of transport such as a bus network, a tram network, or walking routes. Each layer consists of sets of vertices $\mathcal{V}$ (locations) and arcs $\mathcal{A}$ (connections between locations). 

The supergraph $\mathcal{G}$ is created by connecting $\mathcal{V}$ between layers $\mathcal{L}$ with switching arcs $\mathcal{A}_s$, forming a connected network stack. Mathematically:

- Vertices: $\mathcal{V} = \mathcal{V}_R \cup \mathcal{V}_L$
- Arcs: $\mathcal{A} = \mathcal{A}_R \cup \mathcal{A}_L \cup \mathcal{A}_S$

The set of travel requests $\mathcal{R}$ is defined as tuples $r = (o, d, \alpha)$, where $\alpha$ is the number of requests from origin $o$ to destination $d \neq o$. The flow of all active users is defined as:

$$
X \in \mathbb{R}^{|\mathcal{A}| \times |\mathcal{V}|}, \quad X := [x^1, x^2, \dots, x^{|\mathcal{V}|}]
$$

Here, travel requests are grouped by origin (flow bundling), reducing the number of variables required to describe flows in the system. To extract the total flow, an incidence matrix $B$ is defined as:

$$
B_{ia} = 
\begin{cases} 
1 & \text{if flow from arc } a \text{ is directed towards node } i, \\
-1 & \text{if flow from arc } a \text{ is directed away from node } i, \\
0 & \text{otherwise.}
\end{cases}
$$

Additionally, the vector $t$ indicates the travel times $t_a$ between the arcs $a \in \mathcal{A}$.

---

## Rebalancing

For MaaS and taxi-hailing services, vehicles must sometimes travel from areas of low demand to areas of high demand. This rebalancing flow is denoted by $x^r \in \mathbb{R}^{|\mathcal{A}|}$. The cost associated with rebalancing is weighted by a factor $\rho$, representing the cost per unit time of transporting users versus deadheading.

The multi-commodity network is defined as:

$$
\begin{aligned}
\min_{X, x^r} \; J(X, x^r) &= t^\top \left( X1 + \rho x^r \right) \\
\text{s.t.} \quad BX &= D, \\
B(X1 + x^r) &= 0, \\
X, x^r &\geq 0,
\end{aligned}
$$

### Demand Matrix

The demand matrix $D$ is formulated as:

$$
D_{ij} =
\begin{cases} 
\alpha_m, & \exists m \in \mathcal{M} : o_m = j \land d_m = i, \\
-\sum_{k \neq j} D_{kj}, & i = j, \\
0, & \text{otherwise.}
\end{cases}
$$

---

## Ride-Pooling

To solve the ride-pooling assignment, the demand matrix $D$ is modified into the ride-pooling demand matrix $D^{rp}$, accounting for:

1. **Vehicle Capacity**: $k < K$
2. **User Waiting Time**: $t < \overline{t}$
3. **Additional Time Spent Due to Pooling**: $\delta < \overline{\delta}$

Requests are pooled to minimize the network problem cost while satisfying these constraints. Initially, $D$ is calculated by setting $\rho = 0$ to simulate a minimum travel time problem.

### Spatial Feasibility

For a group $\mathcal{C}$ of requests, the delay experienced by request $m \in \mathcal{C}$ is:

$$
\delta_{\mathcal{C}, s}^m = \sum_{p \in \pi_{\mathcal{C}, s}^m} [t^\top X^{\mathcal{C}, s}]_p - t_0^m,
$$

where $\pi_{\mathcal{C}, s}^m$ represents the sequence of nodes traversed for request $m$, and $t_0^m$ denotes the travel time without pooling. Spatial feasibility is met if $\delta_{\mathcal{C}, s}^m \leq \overline{\delta}$ for all $m \in \mathcal{C}$.

### Temporal Feasibility

Using a Poisson process, the pooling probability for $k$ requests in a group is:

$$
P_{\overline{t}}(\alpha_1, \dots, \alpha_k) = \sum_{i=1}^k \frac{\alpha_i}{\sum_{j=1}^k \alpha_j} \prod_{j=1, j \neq i} \left( 1 - e^{-\alpha_j \overline{t}} \right),
$$

where $\alpha_i$ represents the arrival rate of request $i$.

### Ride-Pooling Demand Matrix

The ride-pooling demand matrix $D_{ij}^{rp}$ is:

$$
D_{ij}^{rp} =
\begin{cases} 
\alpha_m + \sum\limits_{\mathcal{C} \in \bigcup_k \mathcal{C}_k^{\overline{\delta}}(M): m \notin \mathcal{C}} \gamma_C D_{\mathcal{C}, \star}^{ij}, & \exists m \in M : (d_m, o_m) = (i, j), \\ \\
-\sum_{k \neq j} D_{kj}^{rp}, & i = j, \\ \\
\sum\limits_{C \in \bigcup_k \mathcal{C}_k^{\overline{\delta}}(M)} \gamma_\mathcal{C} D_{\mathcal{C}, \star}^{ij}, & \text{otherwise}.
\end{cases}
$$

Here, $\gamma_\mathcal{C}$ represents the effective pooled demand for group $\mathcal{C}$, and $D_{\mathcal{C}, \star}^{ij}$ denotes the optimal demand matrix for the group.

This framework ensures computational efficiency and optimal ride-pooling assignments for large-scale networks.

# I-AMoD


# Implementation
