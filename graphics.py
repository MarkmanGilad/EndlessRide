import pygame
from Environment import *
#from game import game
# Colors
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
BLACK = (0,0,0)
BUTTON_COLOR = (50, 50, 255)  # Blue button color
BUTTON_HOVER_COLOR = (100, 100, 255)  # Button hover color
class Background:
    def __init__(self, width, height):
        self.width = width
        self.height = height-100
        self.down = 0

        pygame.font.init()
        self.display = pygame.display.set_mode((width,height))#+100
        self.header_surf = pygame.Surface((width, 100))
        self.surface = pygame.Surface((width, height))
        self.header_surf.fill(WHITE)
        self.header_rect = pygame.Rect(0, 0, width, 100)
        self.surface.fill(GRAY)
        self.draw_dashed_lines()
      
        
        pygame.display.set_caption('Endless Road')
        

    def draw_dashed_lines(self):
        dash_length = 15
        dash_width = 5
        lane_width = 80
        

        for x in range(lane_width, self.width, lane_width):
            self.draw_dashed_line((x, self.down), (x, self.height), dash_width, dash_length)

    def draw_dashed_line(self, start_pos, end_pos, width, dash_length):
        x1, y1 = start_pos
        x2, y2 = end_pos

        if x1 == x2:  # Vertical dashed line
            y_coords = list(range(y1, y2, dash_length * 2))
            for y in y_coords:
                pygame.draw.line(self.surface, WHITE, (x1, y), (x2, y + dash_length), width)
    
    def write (self,surface, text, pos = (50, 20)):
        font = pygame.font.SysFont("arial", 36)
        text_surface = font.render(text, True, BLACK, WHITE)
        surface.blit(text_surface, pos)

    def draw_surface (self):
        self.down = (self.down + 1) % 30
        width, height = self.width, self.height
        self.surface = pygame.Surface((width, height))
        self.surface.fill(GRAY)
        self.draw_dashed_lines()

    # def render(self, env):
    #     #self.good=Environment.score
    #     self.draw_surface()
    #     self.display.blit(self.header_surf, (0, 0))
    #     self.display.blit(self.surface, (0, 100))
    #     self.write (surface=self.header_surf, text="Score: " + str(env.score))
    
    #     for obstacle in env.obstacles_group:
    #         if obstacle.rect.colliderect(self.header_rect):
    #             obstacle.visible = False
    #         else:
    #             obstacle.visible = True
    #     env.car.draw(self.display)
    #     env.obstacles_group.draw(self.display)
    #     env.good_points_group.draw(self.display)  
    def render(self, env):
        self.draw_surface()  # Redraw scrolling background
       
        # Draw the score
        self.write(surface=self.header_surf, text="Score: " + str(env.score))

        # Draw obstacles and good points with an offset
        for obstacle in env.obstacles_group:
            self.surface.blit(obstacle.image, (obstacle.rect.x, obstacle.rect.y))
        
        for good_point in env.good_points_group:
            self.surface.blit(good_point.image, (good_point.rect.x, good_point.rect.y))

        # Draw car without any offset
        env.car.draw(self.surface)
        self.display.blit(self.header_surf, (0, 0))  # Draw the header
        self.display.blit(self.surface, (0, 100))  # Draw the main play surface

    def end_screen(self):
        # Draw the background for the end screen (gray or any other color)
        

        # Create a white square in the middle of the screen
        square_width = 400
        square_height = 400
        square_rect = pygame.Rect((self.width // 2 - square_width // 2, self.height // 2 - square_height // 2), (square_width, square_height))
        pygame.draw.rect(self.display, WHITE, square_rect)

        # Load and position the image (assuming you have a "image.png" file)
        image = pygame.image.load('pics\gameover.png').convert_alpha()  # Replace with the actual image path

        # Scale the image to fit inside the square if needed (optional)
        #image = pygame.transform.scale(image, (350, 350))  # Scale to 100x100 px

        # Position the image at the top center of the square
        image_rect = image.get_rect(center=(square_rect.centerx, square_rect.top + image.get_height() // 2 ))

        # Draw the image
        self.display.blit(image, image_rect)

        # Draw the "Play Again" and "Quit" buttons
        play_again_button = pygame.Rect(self.width // 2 - 150, self.height // 2 + 50, 300, 50)
        quit_button = pygame.Rect(self.width // 2 - 150, self.height // 2 + 120, 300, 50)

        # Draw buttons
        pygame.draw.rect(self.display, BUTTON_COLOR, play_again_button)
        pygame.draw.rect(self.display, BUTTON_COLOR, quit_button)

        # Write button text
        font = pygame.font.SysFont("arial", 36)
        play_again_text = font.render("Play Again", True, WHITE)
        quit_text = font.render("Quit", True, WHITE)

        self.display.blit(play_again_text, (self.width // 2 - play_again_text.get_width() // 2, self.height // 2 + 55))
        self.display.blit(quit_text, (self.width // 2 - quit_text.get_width() // 2, self.height // 2 + 130))

        # Update the display
        pygame.display.flip()

        # Event handling for buttons
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        # Check if the mouse clicks on any button
        if play_again_button.collidepoint(mouse_pos):
            if mouse_pressed[0]:  # Left click
                return 1
                
                print(self.score)
                game.loop()


        if quit_button.collidepoint(mouse_pos):
            if mouse_pressed[0]:  # Left click
                pygame.quit()
                exit()  # Call the quit callback function

            
    
    
