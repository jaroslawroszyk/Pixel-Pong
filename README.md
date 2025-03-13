# Pong AI with NEAT

This project implements an AI that plays Pong using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to train two neural networks to control the paddles. The AI improves its performance over time by playing against itself. Once trained, you can test the AI by running it in the game.

## Requirements

Before running the project, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries, including:

- `pygame`
- `neat-python`
- `pickle`

## How to Train the AI

To train the AI, you need to run the `start_neat_training(config)` function. This function will initialize the NEAT population, restore a checkpoint if available, and begin training the neural networks by having them play against each other.

### Function to call for training:

```python
start_neat_training(config)
```

On the **first run**, the population will be initialized with:

```python
p = neat.Population(config)
```

On subsequent runs, you can restore the population using the checkpoint:

```python
p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-13")
```

Make sure the checkpoint file (e.g., `neat-checkpoint-13`) is in the project directory. You can also modify the checkpoint name or configure NEAT as needed.

Once training is complete, the best-performing model will be saved as `winner.pkl` in folder `models/`.

## How to Run AI with the Trained Model

Once the AI has been trained, you can run it to see how well it plays Pong. The trained model will be loaded from `models/winner.pkl` and used to control the right paddle.

### Function to call for testing with the trained AI:

```python
run_ai_test_with_trained_model(config)
```

This function will load the trained model from `models/winner.pkl`, create the game window, and run the game where the AI plays with the trained neural network.

## Game Features

- **AI Training:** Uses NEAT to evolve neural networks that control the paddles.
- **Pong Game:** A simple Pong game where two paddles try to score against each other.
- **Trained AI Testing:** Run the game with the trained AI to test how well it performs.

## Game Window

The game window size is 800x600 pixels and can be adjusted in the code if necessary. The game is rendered using the Pygame library, which you can modify to suit your preferences.

---

### `requirements.txt`

```plaintext
pygame
neat-python
pickle
```