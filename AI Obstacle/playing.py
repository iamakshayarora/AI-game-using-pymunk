from Game_UI import UI
import numpy as np
from nn import neural_net
import pygame

NUM_SENSORS = 3


def play(model):

    car_distance = 0
    game_state = UI.GameState()

    _, state = game_state.frame_step((2))

    # Move.
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))

        # Take action.
        _, state = game_state.frame_step(action)

        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                running = True
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                exit()


if __name__ == "__main__":
    saved_model = 'saved-models/128-128-64-50000-75000.h5'
    model = neural_net(NUM_SENSORS, [128, 128], saved_model)
    running=True
    while running:
        play(model)
        

    