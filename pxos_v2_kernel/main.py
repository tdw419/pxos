import pygame
import moderngl
import numpy as np

class PxOS_Kernel:
    def __init__(self, width=1024, height=1024):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)

        self.ctx = moderngl.create_context()
        self.width = width
        self.height = height

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return

            self.ctx.clear(0.0, 0.0, 0.0, 1.0) # Black background
            pygame.display.flip()

if __name__ == '__main__':
    kernel = PxOS_Kernel()
    kernel.run()
