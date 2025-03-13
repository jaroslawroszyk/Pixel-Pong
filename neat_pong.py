import time
import pygame
from pong import Game
import neat
import os
import pickle


class PongAIController:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, net):

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.game.move_paddle(left=True, up=False)
            if keys[pygame.K_q]:
                run = False

            output = net.activate(
                (
                    self.right_paddle.y,
                    self.ball.y,
                    abs(self.right_paddle.x - self.ball.x),
                )
            )
            decision = output.index(max(output))
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            self.game.draw(draw_score=True, draw_hits=False)
            pygame.display.update()

        pygame.quit()

    def train_ai_agents(self, genome1, genome2, config, draw=False):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        self.genome1 = genome1
        self.genome2 = genome2
        max_hits = 50

        run = True
        start_time = time.time()

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            game_info = self.game.loop()
            self.move_ai_paddles(net1, net2)

            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            if (
                game_info.left_score >= 1
                or game_info.right_score >= 1
                or game_info.left_hits > max_hits
            ):
                self.calculate_fitness(game_info, duration)
                break
        return False

    def move_ai_paddles(self, net1, net2):
        players = [
            (self.genome1, net1, self.left_paddle, True),
            (self.genome2, net2, self.right_paddle, False),
        ]

        for genome, net, paddle, left in players:
            output = net.activate((paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            valid = True
            if decision == 0:
                penalty = 0.01
                genome.fitness -= penalty
            elif decision == 1:  # Move up
                valid = self.game.move_paddle(left=left, up=True)
            else:  # Move down
                valid = self.game.move_paddle(left=left, up=False)

            if not valid:
                penalty = 1
                genome.fitness -= penalty

    def calculate_fitness(self, game_info, duration):
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration


def create_game_window():
    width, height = 800, 600
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("PixelPong")

    return window, width, height


def evaluate_genomes(genomes, config):
    window, width, height = create_game_window()

    for i, (genome_id1, genome1) in enumerate(genomes):
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i + 1, len(genomes) - 1) :]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            pong = PongAIController(window, width, height)
            force_quit = pong.train_ai_agents(genome1, genome2, config, draw=True)
            if force_quit:
                quit()


def start_neat_training(config):
    if not os.path.exists("models"):
        os.makedirs("models")

    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-14")
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    generations = 2  # Number of generations to run the training for
    winner = p.run(evaluate_genomes, generations)
    with open("models/winner.pkl", "wb") as f:
        pickle.dump(winner, f)


def test_best_network(config):
    window, width, height = create_game_window()

    with open("models/winner.pkl", "rb") as f:
        winner = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = PongAIController(window, width, height)
    game.test_ai(net)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config/config.txt")

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    start_neat_training(config)
    # test_best_network(config)
