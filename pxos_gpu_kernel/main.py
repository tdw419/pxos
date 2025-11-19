import pygame
import moderngl
import numpy as np

from .ir import IRKernel, IRParameter, IRInstruction, IROperand, IROperation, IRType
from .backend_glsl import GLSLEmitter

class GpuMicrokernel:
    def __init__(self, width=1024, height=1024):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)

        self.ctx = moderngl.create_context()
        self.width = width
        self.height = height

        # VRAM Memory Model
        self.texture_a = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_b = self.ctx.texture((width, height), 4, dtype='f4')

        # Define shader logic using pxIR
        kernel_ir = self._create_inversion_kernel()

        # Compile IR to GLSL
        emitter = GLSLEmitter()
        glsl_code = emitter.emit(kernel_ir)

        print("--- Generated GLSL ---")
        print(glsl_code)
        print("----------------------")

        # Load Compute Shader from generated code
        self.compute_shader = self.ctx.compute_shader(glsl_code)

    def _create_inversion_kernel(self) -> IRKernel:
        """Creates the IR for a simple color inversion shader."""

        # Parameters (uniforms)
        tex_in = IRParameter(name="texture_in", ir_type=IRType.IMAGE2D, binding=0, qualifiers=["readonly"])
        tex_out = IRParameter(name="texture_out", ir_type=IRType.IMAGE2D, binding=1, qualifiers=["writeonly"])

        # Operands (variables and literals)
        texel_coord = IROperand(name="ivec2(gl_GlobalInvocationID.xy)", ir_type=IRType.INT2, is_literal=True)
        color = IROperand(name="color", ir_type=IRType.FLOAT4)
        one_vec3 = IROperand(name="vec3(1.0)", ir_type=None, is_literal=True)
        color_rgb = IROperand(name="color.rgb", ir_type=None)
        inverted_rgb = IROperand(name="inverted_rgb", ir_type=None)
        color_a = IROperand(name="color.a", ir_type=None)
        final_color = IROperand(name="final_color", ir_type=IRType.FLOAT4)

        # Instructions
        instructions = [
            IRInstruction(op=IROperation.LOAD_IMAGE, operands=[tex_in, texel_coord], result=color),
            IRInstruction(op=IROperation.SUBTRACT, operands=[one_vec3, color_rgb], result=inverted_rgb),
            IRInstruction(op=IROperation.CONSTRUCT_FLOAT4, operands=[inverted_rgb, color_a], result=final_color),
            IRInstruction(op=IROperation.STORE_IMAGE, operands=[tex_out, texel_coord, final_color])
        ]

        return IRKernel(name="invert_kernel", parameters=[tex_in, tex_out], instructions=instructions)


    def run(self):
        # Initialize texture_a with a pattern
        buffer_data = np.zeros((self.width, self.height, 4), dtype='f4')
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)
        buffer_data[:, :, 0] = xv      # R
        buffer_data[:, :, 1] = yv      # G
        buffer_data[:, :, 2] = 0.5     # B
        buffer_data[:, :, 3] = 1.0     # A
        self.texture_a.write(buffer_data.tobytes())

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return

            # Bind textures for compute shader
            self.texture_a.bind_to_image(0, read=True, write=False)
            self.texture_b.bind_to_image(1, read=False, write=True)

            # Run compute shader
            self.compute_shader.run(group_x=self.width // 16, group_y=self.height // 16)

            # Swap textures for next frame
            self.texture_a, self.texture_b = self.texture_b, self.texture_a

            # Blit the result to the screen
            fbo = self.ctx.framebuffer(color_attachments=[self.texture_a])
            self.ctx.copy_framebuffer(self.ctx.screen, fbo)

            pygame.display.flip()

if __name__ == '__main__':
    kernel = GpuMicrokernel()
    kernel.run()
