import pygame
import torch
import random
import utils.constants as constants
import os



class OrganismModel:
    def __init__(self, gravity):
        self.model = self.design_organism()
        self.joint_angles = torch.zeros(2)  # Initialize two joint angles
        self.position = torch.tensor([0.0, 0.0])  # Position in 2D space
        self.velocity = torch.tensor([0.0, 0.0])  # Velocity in 2D space
        self.gravity = gravity  # Initialize gravity

    def design_organism(self):
        # Initialize organism with random parameters
        return torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )

    def forward(self, x):
        # Simulate movement based on joint angles
        movement = torch.sum(self.joint_angles).item()
        # Update velocity and position
        self.velocity += torch.tensor([movement, -self.gravity])
        self.position += self.velocity
        return self.position[0].item()  # Forward movement

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)

    def fitness(self, data):
        # Calculate fitness based on reaching the goal
        final_x = self.position[0].item()
        fitness_score = final_x  # Higher x position is better
        return fitness_score

    def mutate(self):
        # Mutate model parameters and joint angles
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
            self.joint_angles.add_(torch.randn_like(self.joint_angles) * 0.05)

    def train(self, data):
        # Train using a simple evolutionary algorithm
        for generation in range(100):
            fitness = self.fitness(data)
            self.mutate()
            print(f"Generation {generation}: Fitness = {fitness}")

    def test(self, data):
        # Test the organism on new data
        fitness = self.fitness(data)
        print(f"Test Fitness: {fitness}")

    def draw(self, screen):
        # Visualize the organism with two joints
        total_param_sum = sum(param.abs().sum().item() for param in self.model.parameters())
        size = int(total_param_sum) % 100 + 10
        x, y = int(self.position[0].item()) + screen.get_width() // 4, screen.get_height() // 2
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 10)
        joint_length = 50
        angle1, angle2 = self.joint_angles
        joint1_pos = (x + int(joint_length * torch.cos(angle1).item()), y + int(joint_length * torch.sin(angle1).item()))
        joint2_pos = (joint1_pos[0] + int(joint_length * torch.cos(angle2).item()), joint1_pos[1] + int(joint_length * torch.sin(angle2).item()))
        pygame.draw.line(screen, (0, 0, 255), (x, y), joint1_pos, 5)
        pygame.draw.line(screen, (0, 0, 255), joint1_pos, joint2_pos, 5)

    def update_physics(self):
        # Apply gravity
        self.velocity += torch.tensor([0.0, -self.gravity])
        self.position += self.velocity

class OrganismDesigner:
    def __init__(self,device):
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Organism Designer')
        button_top = 600 - 50
        button_left = 50
        button_width = 150  # Increased button width for better spacing
        button_spacing = 20  # Added spacing between buttons
        self.stepButton = (pygame.Rect(button_left + (button_width + button_spacing) * 0, button_top, button_width, 50), 'Simulate Step')
        self.trainButton = (pygame.Rect(button_left + (button_width + button_spacing) * 1, button_top, button_width, 50), 'Train Model')
        self.testButton = (pygame.Rect(button_left + (button_width + button_spacing) * 2, button_top, button_width, 50), 'Test Model')
        self.resetButton = (pygame.Rect(button_left + (button_width + button_spacing) * 3, button_top, button_width, 50), 'Reset')
        self.saveButton = (pygame.Rect(button_left + (button_width + button_spacing) * 4, button_top, button_width, 50), 'Save')
        self.loadButton = (pygame.Rect(button_left + (button_width + button_spacing) * 5, button_top, button_width, 50), 'Load')
        self.buttons = [
            self.stepButton,
            self.trainButton,
            self.testButton,
            self.resetButton,
            self.saveButton,
            self.loadButton
        ]

        # Environment parameters
        self.gravity = 9.81
        self.ground_level = self.screen.get_height() - 50
        self.goal_position = self.screen.get_width() - 100

        self.model = OrganismModel(self.gravity)  # Pass gravity to OrganismModel

        self.model.to(device)

        pygame.font.init()
        self.font = pygame.font.SysFont('Comic Sans MS', 30)

    def check_button(self, pos):
        for button in self.buttons:
            if button[0].collidepoint(pos):
                return button
            
    def draw_buttons(self):
        for button in self.buttons:
            pygame.draw.rect(self.screen, (70, 130, 180), button[0])  # Changed color for better visibility
            text = self.font.render(button[1], True, (255, 255, 255))
            text_rect = text.get_rect(center=button[0].center)
            self.screen.blit(text, text_rect)

    def draw_environment(self):
        # Draw ground
        pygame.draw.rect(self.screen, (139, 69, 19), pygame.Rect(0, self.ground_level, self.screen.get_width(), 50))
        # Draw goal
        pygame.draw.rect(self.screen, (255, 215, 0), pygame.Rect(self.goal_position, self.ground_level - 100, 50, 100))

    def reset(self):
        print('Reset')
        self.model.position = torch.tensor([0.0, 0.0], dtype=torch.float)
        self.model.velocity = torch.tensor([0.0, 0.0], dtype=torch.float)
        self.model.joint_angles = torch.zeros(2)
        print('Organism has been reset to initial state.')

    def save_model(self):
        print('Save')
        path = 'organism_model.pth'
        self.model.save(os.path.join(constants.MODEL_DIR, path))
        print(f'Model saved to {path}.')

    def load_model(self):
        print('Load')
        path = 'organism_model.pth'
        self.model.load(os.path.join(constants.MODEL_DIR, path))
        print(f'Model loaded from {path}.')

    def step(self):
        print('Simulate Step')
        x = torch.randn(4)
        y = self.model.forward(x)
        print(f'Position X: {y}')
        # Check for collision with ground
        if self.model.position[1].item() >= self.ground_level:
            self.model.position[1] = torch.tensor(self.ground_level, dtype=torch.float)
            self.model.velocity[1] = torch.tensor(0.0)
        # Check if goal is reached
        if self.model.position[0].item() >= self.goal_position - 50:
            print('Goal Reached!')

    def train(self):
        print('Train')
        data = torch.randn(100, 4)
        self.model.train(data)


    def test(self):
        print('Test')
        data = torch.randn(100, 4)
        self.model.test(data)



    def run_gui(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    button = self.check_button(pos)
                    if button:
                        if button == self.stepButton:
                            self.step()
                        elif button == self.trainButton:
                            self.train()
                        elif button == self.testButton:
                            self.test()
                        elif button == self.resetButton:
                            self.reset()
                        elif button == self.saveButton:
                            self.save_model()
                        elif button == self.loadButton:
                            self.load_model()

            # Apply continuous physics
            self.model.update_physics()

            # Handle collision with ground
            if self.model.position[1].item() >= self.ground_level:
                self.model.position[1] = torch.tensor(self.ground_level, dtype=torch.float)
                self.model.velocity[1] = torch.tensor(0.0)

            # Check if goal is reached
            if self.model.position[0].item() >= self.goal_position - 50:
                print('Goal Reached!')

            self.screen.fill((135, 206, 235))  # Sky blue background
            self.draw_environment()
            self.model.draw(self.screen)
            # Draw buttons
            self.draw_buttons()

            # Update the display
            pygame.display.flip()

        pygame.quit()

    

if __name__ == '__main__':
    designer = OrganismDesigner(constants.DEVICE)
    designer.run_gui()