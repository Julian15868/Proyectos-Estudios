import pygame


# Vars
grid_size = 9
goal_position = (7, 2)  # Posición de estacionamiento
block_positions = [(7, 1), (6, 2), (7, 3)]  # Bloques a evitar
actions = ['up', 'down', 'left', 'right']
action_size = len(actions)

#Funcs
def step(state, action):
    x, y = state
    if action == 'up' and x > 0:
        x -= 1
    elif action == 'down' and x < grid_size - 1:
        x += 1
    elif action == 'left' and y > 0:
        y -= 1
    elif action == 'right' and y < grid_size - 1:
        y += 1
    next_state = (x, y)
    
    if next_state == goal_position:
        reward = 1
    elif next_state in block_positions:
        reward = -2
    else:
        reward = 0
    return next_state, reward

# Función para elegir una acción usando epsilon-greedy
def choose_action(state):
    if np.random.rand() < epsilon:
        return random.choice(actions)  # Explorar
    else:
        q_values = model.predict(np.array([state]))[0]  # Explotar usando la red neuronal
        return actions[np.argmax(q_values)]


# Inicializar Pygame después del entrenamiento
pygame.init()
window_size = 500
cell_size = window_size // grid_size
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("Aprendizaje por Refuerzo - Estacionamiento de Nave")

# Colores
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Función para dibujar la cuadrícula
def draw_grid():
    screen.fill(WHITE)
    for x in range(0, window_size, cell_size):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, window_size))
        pygame.draw.line(screen, (200, 200, 200), (0, x), (window_size, x))
    pygame.draw.rect(screen, GREEN, (goal_position[1] * cell_size, goal_position[0] * cell_size, cell_size, cell_size))
    for block in block_positions:
        pygame.draw.rect(screen, RED, (block[1] * cell_size, block[0] * cell_size, cell_size, cell_size))

# Simulación con Pygame
running = True
state = (0, 0)
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_grid()
    pygame.draw.rect(screen, BLUE, (state[1] * cell_size, state[0] * cell_size, cell_size, cell_size))
    pygame.display.update()
    
    # Elegir la mejor acción según la red neuronal entrenada
    action = choose_action(state)
    next_state, _ = step(state, action)
    state = next_state
    
    # Controlar la velocidad del juego
    clock.tick(5)  # Ajusta la velocidad a 5 FPS para movimientos más visibles
    
    # Reiniciar si la nave llega a la meta
    if state == goal_position:
        time.sleep(1)
        state = (0, 0)  # Reiniciar la posición de la nave

pygame.quit()
