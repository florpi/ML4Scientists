
Assignments for NYU's graduate class on Machine Learning for Scientists PHYS-GA 2043 (more information can be found at https://florpi.github.io/ml-for-scientists.html). The goal is to develop a solid foundation and understanding in Machine Learning, so many of the assignments involve implementing different methods from scratch.

# A1) Training a multilayer perceptron from scratch, in JAX

A tutorial on what is actually inside `nn.Sequential` and `torch.optim.Adam`.

You build an MLP from nothing but arrays — the parameters, the forward pass, optimizers, and the training
loop. The one thing you do *not* write from scratch is backpropagation; `jax.grad` handles that.

The optimizers (SGD, momentum, Adagrad, Adam, Muon) are compared on a regression problem: a sum
of sines at frequencies 1, 2, 4 and 8, equal amplitude. 

# Lecture notes

The [`notes/`](notes) folder has the lecture notes, as PDFs with their LaTeX source:

- **Lecture 1 — MLPs and optimisation:** the supervised learning problem, perceptrons, MLPs, universal approximation, and optimisers (GD, momentum, SGD, AdaGrad, RMSProp, Adam, Muon).
- **Lecture 2 — Backpropagation and losses:** autodiff and computational graphs, backpropagation, vanishing/exploding gradients, residual connections, and choosing a loss.

Found a mistake or something unclear? Please [open an issue](https://github.com/florpi/ML4Scientists/issues) or submit a pull request with a fix to the `.tex` file.

# Branches

- **`main`** — the assignments, with the pieces you implement left blank.
- **`solutions`** — the complete, working versions (released after assignments are due).

