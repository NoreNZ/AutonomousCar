import random
import sys
import os
import neat
import pygame
import math

WIN_WIDTH = 1000
WIN_HEIGHT = 800

CAR_SIZE_X = 30   
CAR_SIZE_Y = 30

COLLIDE_COLOUR = (255, 255, 255, 255) # Colour on map to collide with

current_gen = 0 # Generation counter
change_map = 1
drawradars = True
bestcarfitness = 1
oldbestcarfitness = 1
gamespeed = 60

class Car:

    def __init__(self):
        # Car sprite
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 

        self.position = [470, 145] # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False

        # Calculate centre of car
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        # Radars
        self.radars = []
        self.drawing_radars = []
        # Check if car crashed 
        self.alive = True

        # Distance and time counters for reward calculation
        self.distance = 0
        self.time = 0 

    # Draw car and radars function
    def draw(self, screen):
        # Draw rotated car sprite
        screen.blit(self.rotated_sprite, self.position)
        # Draw car radars
        if drawradars == True:
            for radar in self.radars:
                position = radar[0]
                pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
                pygame.draw.circle(screen, (0, 255, 0), position, 3)

    def rotate_center(self, image, angle):
        # Rotate car sprite
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If any corner touches collide colour, crash
            if game_map.get_at((int(point[0]), int(point[1]))) == COLLIDE_COLOUR:
                self.alive = False
                break
    def update_collision(self):
        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate cars corner
        length = 0.5 * CAR_SIZE_X
        LT = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        RT = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        LB = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        RB = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [LT, RT, LB, RB]

    def update_sprite(self):
        # Update sprite rotation
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        # Bind sprite to X boundries
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIN_WIDTH - 20)
        # Bind sprite to Y boundaries
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIN_WIDTH - 20)

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While radar doesn't hit collision colour and length is less than value, keep extending radar length
        while not game_map.get_at((x, y)) == COLLIDE_COLOUR and length < 20 * self.speed:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate radar distance and append to radars list
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # if starting speed not set, set speed to 20 and update speed_set
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        #Update sprite rotation and location
        self.update_sprite()

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        #Update sprite center and corner coordinates
        self.update_collision()

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # Check radars betwwen -90 and 120 increment 45 degrees
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get radar measurements
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)
        # Return radar measurements
        return return_values

    def is_alive(self):
        # Return car alive state
        return self.alive

    def get_reward(self):
        # return reward
        return self.distance


def run_sim(genomes, config):
    
    # Neural network collection
    nets = []
    # Car collection
    cars = []

    # Initialize PyGame and display
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    # For all genomes create a neural network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        # Add car to car list
        cars.append(Car())

    clock = pygame.time.Clock()
    # Font settings
    font = pygame.font.SysFont("Arial", 20)
    fontsmall = pygame.font.SysFont("Arial", 15)
    # Call global variables
    # Generation counter
    global current_gen
    # Map selection variable
    global change_map
    # Toggle radar rendering variable
    global drawradars
    # Measure best car fitness
    global bestcarfitness
    # Game speed (in FPS)
    global gamespeed
    # Default map to draw
    if change_map == 1:
        game_map = pygame.image.load('map.png').convert()
    # Map 2
    if change_map == 2:
        game_map = pygame.image.load('map1.png').convert()
    # Map 3
    if change_map == 3:
        game_map = pygame.image.load('map2.png').convert()
    # Map 4
    if change_map == 4:
        game_map = pygame.image.load('map3.png').convert()
    # Increment generation counter
    current_gen += 1

    # Counter to measure time passed, for capping simulation time
    counter = 0

    while True:
        # Check for keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                # Exit program
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                # End generation
                if event.key == pygame.K_SPACE:
                    return
                # Select map 1
                if event.key == pygame.K_1:
                    change_map = 1
                # Select map 2
                if event.key == pygame.K_2:
                    change_map = 2
                # Select map 3
                if event.key == pygame.K_3:
                    change_map = 3
                # Select map 4
                if event.key == pygame.K_4:
                    change_map = 4
                # Toggle radar render
                if event.key == pygame.K_r:
                    drawradars = not drawradars
                # Increase game speed
                if event.key == pygame.K_EQUALS:
                    if gamespeed <= 150:
                        gamespeed += 30
                if event.key == pygame.K_MINUS:
                    if gamespeed >= 60:
                        gamespeed -= 30


        # Iterate through cars and get action, 0,1,2 or no action options
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 15 # Left
            elif choice == 1:
                car.angle -= 15 # Right
            elif choice == 2:
                if(car.speed - 2 >= 5):
                    car.speed -= 5 # Slow Down
            else:
                car.speed += 2 # Speed Up
        

        # If car still alive, update fitness function
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            oldbestcarfitness = bestcarfitness
            break

        # Increment counter
        counter += 1
        #If counter reaches max limit, break (time limit)
        if counter == 1200:
            oldbestcarfitness = bestcarfitness
            break

        # Draw map and alive cars
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display details
        text = font.render("Generation: " + str(current_gen), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.topleft = (0, 0)
        screen.blit(text, text_rect)

        text = font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (500, 10)
        screen.blit(text, text_rect)

        text = font.render("Map Selection: " + str(change_map), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (840, 0)
        screen.blit(text, text_rect)

        text = font.render("Radar draw: " + str(drawradars), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (840, 20)
        screen.blit(text, text_rect)

        for i, car in enumerate(cars):
            if car.is_alive():
                if genomes[i][1].fitness > bestcarfitness:
                    bestcarfitness = genomes[i][1].fitness

        text = font.render("Best fitness: " + str(bestcarfitness), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (0, 20)
        screen.blit(text, text_rect)

        text = font.render("Game timescale: " + str(gamespeed/60) + "x", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topright = (995, 40)
        screen.blit(text, text_rect)

        text = fontsmall.render("Number 1-4 to change map" , True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.bottomleft = (0, 780)
        screen.blit(text, text_rect)

        text = fontsmall.render("CONTROLS | ESCAPE - End simulation | SPACEBAR - End generation | R - Toggle radar draw | MINUS - Reduce game speed | PLUS - Increase game speed" , True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.bottomleft = (0, 800)
        screen.blit(text, text_rect)


        pygame.display.flip()
        clock.tick(gamespeed) # GAME SPEED

if __name__ == "__main__":
    
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Max 1000 generations
    population.run(run_sim, 1000)
