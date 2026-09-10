# Training a multilayer perceptron from scratch, in JAX

A tutorial on what is actually inside `nn.Sequential` and `torch.optim.Adam`.

You build an MLP from nothing but arrays — the parameters, the forward pass, five optimizers, and the training
loop. The one thing you do *not* write from scratch is backpropagation; `jax.grad` handles that, and the point
of the notebook is everything around it.

The optimizers (SGD, momentum, Adagrad, Adam, Muon) are compared on a **multi-scale regression** target: a sum
of sines at frequencies 1, 2, 4 and 8, equal amplitude. Because networks learn smooth structure long before
sharp structure, this target has a built-in difficulty gradient — plain gradient descent learns the slowest
wave and stalls, while better optimizers climb further up the ladder. A per-frequency diagnostic shows exactly
how far each one gets.

The notebook ends with the same model in ~10 lines of PyTorch, and a table mapping every abstraction back to
the low-level code it replaces.

## Running it

Open `1-mlp-from-scratch.ipynb` in Google Colab and uncomment the install line in the first cell. It runs on
CPU in a couple of minutes — no GPU needed. Locally, you need `jax`, `matplotlib` and `torch`.

## Branches

- **`main`** — the tutorial, with the pieces you implement left blank.
- **`solutions`** — the complete, working version.

Get stuck, peek at `solutions`, but write it yourself first. Ask the teaching assistants.
