# genetic-knapsack
Searching the `knapsack` (an NP) problem space by optimizing some discrete heuristic function using genetic algorithms. 

## overview of GA in problem scope
We aim to solve the knapsack problem with genetic algorithms by viewing the possible
configurations (a `2^n` space) as gene sequences, and some heuristic involving the weight
and value of each item as the objective function to optimize, hoping to find the optimal knapsack configuration for the thief by iteratively evolving populations geenrated gene sequences (bit strings) using crossover and mutation, and recording and optimizing the value of the objective function along the way by carrying the best/optimal trait combinations (characterized by highest value when used as input to the objective heuristic) through several cycles of evolution. Ranking each generation between iterations (evolution cycles) as a process has some properties:
  1) it has only knowledge of a fixed-size population, not the entire NP search space
  2) it is concerned only with the local optima (that which can be found in the current generation) and does not backtrack
therefore the process of scoring and ranking in order to emulate natural selection (the nature of GAs) is both uninformed and greedy due to 1 and 2 respectively. It does howevever "prune" the search space by cutting the population size by some factor between iterations, but quickly replaces the lost genetic sequences by "breeding the k-best scored sequences" left over from the reduction. because the population size remains constant, the problem becomes of maximizing the objective function and finding an optimal solution given some upperbound on iteration becomes polynomial, whereas finding the best remains NP. This is sufficient, as we are concerned with optimizing things. That is, so long as our solution is shown to get better with time and opportunity, we do not have to show it is the best. 

## big picture
In terms of ML, genetic algorithms are a form of unsupervised learning. Given a set of traits and their cost/value, we can reach an optima (defined as such by a threshold) without any other information. You can view them as a discrete treatment of optimization problems that allows you to build stochastic/informed-random heuristics intuitively and creatively. Similar to optimizing Regression betas (a non-poylnomial search space) with continuous Gradient boosting, we can optimize 'gene sequences' (see above text for explanation) with stochastic 'evolutionary' epochs. There is some research that, given some mapping of the involved response and observation vectors to bit strings (an encoding), you could discetely optiize continuous problems using GAs. 

### 1. Fitness Functions
The fitness of a gene sequence represents the metric of the objective, and is therefore the (in this case discrete) heuristic function to iteratively optimize. These functions, $f$ and $g$, are cocnerned with the set of all values encoded for by the bitstring, $\mathbb V$, and the set of all weights encoded for by the bitstring $\mathbb W$. It is constrained given maximum allowable cost $m$. Note that $\mathbb W$ and $\mathbb V$ are calculated by adding a value and/or weight to their respective set if the corresponding bit in the bitstring is a logic `1`. This process is completed by the encoding function $h$. (note: in code, $h$ occurs implicitly within the runtime of $f$ and $g$).

### 1.1 Definitions
$h$ takes as input a binary gene sequence and gives the sets $\mathbb W$ and $\mathbb V$ respectively. 
```math
h(x) \rightarrow \mathbb W,\mathbb V \text{ }\text{where }\text{ } \forall i\in\mathbb \{0,...,n\}, (\mathbb W_i,\mathbb V_i)\in\mathbb T
```
Where $n$ is the length of the base-2 number $x$ in bits (length of our bit string in code) given by the magnitude of the universal trait set $|\mathbb T|$, a set of n-tuples `(weight, value)`. 
I employ two primary 'fitness functions' as we will call them. The first is that which was presented to us in the assignment specification, $f$. 
```math
    f(\mathbb W, \mathbb V)= 
\begin{cases}
    \sum\limits_{i=1}^n \mathbb V_i,& \text{if } \sum\limits_{i=1}^n \mathbb W_i\leq m\\
    \\
    \\
    -1, & \text{otherwise}
\end{cases}
```
Recall that $m$ is the constraint on the value of the fitness function. The next fitness function I employed in code, $g$, is described below. 
```math
    g(\mathbb W, \mathbb V)= 
\begin{cases}
    \sum\limits_{i=1}^n \left(\dfrac{\mathbb V_i}{\mathbb W_i}\right),& \text{if } \sum\limits_{i=1}^n \mathbb W_i\leq m\\
    \\
    \\
    -1, & \text{otherwise}
\end{cases}
```

### 1.2 Analysis
The provided fitness function $f$ accumulates all encoded-for values and reports this as the score unless the maximum allowable cost $m$ is exceeded. The second fitness function is more relative, as it is progressively weighted in terms of the associated cost ($\mathbb w_i$) of enabling a trait (taking an item and putting it in the "knapsack"). In other words, $h$ is a sum of ratios also constrained by $m$. Two fitness functions will clearly report different optimal bit strings, so comparing them is difficult. Ideally, comparable fitness functions should be somewhat correlated and of the same unit. This way, both the optimal bit string as reported by $f$ and as reported by $h$, however different, should appear in the top $k$ elements of a ranked generation after a given number of evolutionary cycles. This allows us to use one function, $f$, as a means of evaluating a given generation's most fit gene sequence against a threshold $t$ for early convergence, and the better augmented fitness function $h$ to evolve generations between cycles. It follows that using one fitness function or the other to evolve would also not make a signifcant difference in convergence time. Using $f$ to test for the threshold-fitness in a generation is only convenient because the test-trait sets and optimal bit strings given in the assignment were pre-solved using this function. 

## 2. Schema & Threshold
This was by far the most interesting part of the project because of what it taught me about stochastic processes. Naturally, the ranking process (involving the fitness function) combined with the genetic algorithm steps (stochastic genetic cross-over and mutation) makes for sufficient variance between evolutionary generations. The Schema is concerned with creating sufficient genetic variance during in the initial population before any evolution takes place.  Here, schema means some method (random probability, constant, variable function, etc.) by which to generate an initial population with. Multiple approaches were tested, including two of the three listed in parenthesis above were evaluated during the course of this project. 

### 2.1 Threshold
The threshold is a value/output of $f$ or $g$ that allows the discrete optimization ("evolution process") to converge before the maximum number of iterations is reached. Conceptually, this threshold can be thought of as an *uninformed heuristic* (no parameters besides the values and weights of each trait, no knowledge of the bounds on the problem space given it is NP) as it should be some initial guess at what genetic sequence ("knapsack configuration") is optimal. Because it is done before any optimization iterations ("evolutionary epochs") and therefore before any real work is done, it should be just that: Purely an initial guess and computed in either linear $O\(n\)$ or quadratic $O\(n^2\)$ time where $n$ is the size of the trait set $|\mathbb T|$. 

### 2.2 Relationship Between Schema & Threshold
The intersection between provided definitions of schema and threshold is clear: both are concerned with making an initial guess at what is optimal before iterative optima-search comnmences. The difference is that one provides for catalyzation of the algorithms "iterative section" and takes the form of the actual bitstring, while one provides for convergence of the algorithms "iterative section" and takes the form of a value. Can you guess which is which? Schema asks "what is a good initial of the members of the population before I let random probability and genetic operations (i.e. natural selection, a stohcastic combination of the two) take control of optimization?" while the other asks "what is a good guess for what the optimal value is so that I know when to stop evolving generations?". The later is almost paradoxical if not directly contradicting of itself. If you are progressing towards an unknown end, you're finding a needle in an NP-complete haystack such that you don't know if you have truly found a needle, only that this piece of hay looks more "needly" than this piece of hay. Oh and theres a chance it isn't hay, it could be the needle! Do you see the issue? It should be noted that computing a threshold for early convergence is most useful in testing when the optimal value is known ahead of time, unless there is some best estimate for the optimal value that does not involve thousands of heuristic-driven iterations through an expoential search space. Even if the later is the case, this genious polynomial-time estimate should not be used as to terminate optima finding, but to initialize members of the initial population before iterative optima finding commences. With this in mind, we will pursue development of some method to compute threshold and in the process uncover possible Schema.

### 2.3 Developing Solutions for Schema & Threshold 

### 2.4 Population Size & Comparing Schemas (UDATE TO INCLUDE ALL DISCUSSED THRESHOLD/SCHEMA METHODS ABOVE)
On a population size of `1000` with a trait set of magnitude `10` (one of the test sets given in the assignment), the random approach lead to convergence within 0-20 iterations. However, with a smaller population size like `100`, this was not the case even with the same trait set. Giving 'nature' a head start by applying a Schema that switches "on" (sets to logic `1`) the top $k$ traits improved performance in smaller spaces. Interestingly enough, combining this method with randomness (50% chance of remaing $|\mathbb T| - k$ bits to be enabled or disabled) resulted in poor performance (many more iterations to converge to the optimal value). The afforementioned phenomenon is described by the simple fact that a larger population size gives random probability much more "control" over the evolitionary process in that there is a higher chance you will just happen upon an optimal solution early on, especially given that the algorithm strives to keep the population size constant between generations. By shrinking the population size by a factor of 10, the ranking process (fitness functions) and stochastic genetic operations (cross-over and mutation) have a greater opportunity to find the optimal solution throughout evolutionary cycles/optimization iterations. In other words, decreasing the population thereby lesening the involvement of random probability will put a bigger load of the optimization problem on the interchangeable elements of the evolution process making a comparison of those elements more informative. Comparing schemas is but one example of this idea in affect at the first stage. This applies to the ranking process and genetic operations at n-stages of iteration as well. It should be noted, however, if the number of maximum iterations is increased proportional to the decrease in population size you will fail to decrease the influence that random probability has on convergence time.
