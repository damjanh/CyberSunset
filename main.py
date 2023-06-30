import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr


class SimpleComponent:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)


class SentientComponent:
    def __init__(self, position, eulers, health):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.state = 'fallingOn'
        self.can_shoot = True
        self.reload_time = 20
        self.falling_time = 0

    def shoot(self):
        if self.can_shoot and self.state == 'stable':
            print('shoot')
            self.can_shoot = False
            self.reload_time = 5

    def update(self, rate):
        if self.state == 'stable':
            if abs(self.velocity[1]) < 0.01:
                self.eulers[0] *= 0.9
                if abs(self.eulers[0] < 0.5):
                    self.eulers[0] = 0
            else:
                self.position += self.velocity / 4
                self.eulers[0] += 8 * self.velocity[1]
                self.velocity = np.array([0, 0, 0], dtype=np.float32)

                self.position[1] = min(6, max(-6, self.position[1]))
                self.eulers[0] = min(45, max(-45, self.eulers[0]))

            if not self.can_shoot:
                self.reload_time -= rate
                if self.reload_time < 0:
                    self.reload_time = 0
                    self.can_shoot = True

        elif self.state == 'fallingOn':
            self.position[2] = 0.99 + (0.9 ** self.falling_time) * 18
            self.falling_time += rate
            if self.position[2] < 1:
                self.position[2] = 1
                self.state = 'stable'


class Scene:
    def __init__(self):
        self.enemy_spawn_rate = 0
        self.power_ups_spawn_rate = 0
        self.enemy_shoot_rate = 0

        self.player = SentientComponent(
            position=[1, 0, 1],
            eulers=[0, 90, 0],
            health=36
        )

        self.enemies = []
        self.bullets = []
        self.power_ups = []

    def update(self, rate):
        self.player.update(rate)

    def move_player(self, d_pos):
        if self.player.state == 'stable':
            self.player.velocity += d_pos


class App:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.renderer = GraphicsEngine()
        self.scene = Scene()
        self.last_time = pg.time.get_ticks()
        self.current_time = 0
        self.num_frames = 0
        self.frame_time = 0
        self.light_count = 0

        self.main_loop()

    def main_loop(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False

            self.handle_keys()

            self.scene.update(self.frame_time * 0.05)

            self.renderer.render(self.scene)

            self.calculate_frame_rate()

        self.quit()

    def handle_keys(self):
        keys = pg.key.get_pressed()
        rate = self.frame_time / 16

        if keys[pg.K_LEFT]:
            self.scene.move_player(rate * np.array([0, 1, 0], dtype=np.float32))
        elif keys[pg.K_RIGHT]:
            self.scene.move_player(rate * np.array([0, -1, 0], dtype=np.float32))

        if keys[pg.K_SPACE]:
            self.scene.player.shoot()

    def calculate_frame_rate(self):
        self.current_time = pg.time.get_ticks()
        delta = self.current_time - self.last_time
        if delta >= 1000:
            frame_rate = max(1, int(1000.0 * self.num_frames/delta))
            pg.display.set_caption(f'Running at {frame_rate} fps.')
            self.last_time = self.current_time
            self.num_frames = -1
            self.frame_time = float(1000.0 / max(1, frame_rate))
        self.num_frames += 1

    def quit(self):
        self.renderer.destroy()


class GraphicsEngine:
    def __init__(self):
        # colors
        self.palette = {
            'Navy': np.array([0, 13/255, 107/255], dtype=np.float32),
            'Purple': np.array([156/255, 25/255, 244/255], dtype=np.float32),
            'Pink': np.array([225/225, 93/255, 162/255], dtype=np.float32),
            'Teal': np.array([153/225, 221/255, 204/255], dtype=np.float32)
        }

        # init pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640, 480), pg.OPENGL | pg.DOUBLEBUF)

        # init OpenGL
        glClearColor(self.palette['Navy'][0], self.palette['Navy'][1], self.palette['Navy'][2], 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = self.create_shader('shaders/vertex.txt', 'shaders/fragment.txt')
        self.render_pass = RenderPass(shader)
        self.mountain_mesh = Mesh('models/mountains.obj')
        self.grid_mesh = Grid(48)
        self.player_mesh = Mesh('models/rocket.obj')

    def create_shader(self, vertex_file_path, fragment_file_path):
        with open(vertex_file_path, 'r') as f:
            vertex_src = f.readlines()

        with open(fragment_file_path, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

        return shader

    def render(self, scene):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.render_pass.render(scene, self)

        pg.display.flip()

    def destroy(self):
        pg.quit()


class RenderPass:
    def __init__(self, shader):
        # init OpenGL
        self.shader = shader
        glUseProgram(self.shader)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=800/600,
            near=0.1, far=100, dtype=np.float32
        )

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, 'projection'),
            1, GL_FALSE, projection_transform
        )

        self.model_matrix_location = glGetUniformLocation(self.shader, 'model')
        self.view_matrix_location = glGetUniformLocation(self.shader, 'view')
        self.color_location = glGetUniformLocation(self.shader, 'object_color')

    def render(self, scene, engine):
        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye=np.array([-10, 0, 4], dtype=np.float32),
            target=np.array([1, 0, 4], dtype=np.float32),
            up=np.array([0, 0, 1], dtype=np.float32),
            dtype=np.float32
        )
        glUniformMatrix4fv(self.view_matrix_location, 1, GL_FALSE, view_transform)

        # mountains
        glUniform3fv(self.color_location, 1, engine.palette['Teal'])
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_z_rotation(theta=np.radians(90), dtype=np.float32)
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(vec=np.array([32, 0, 0], dtype=np.float32))
        )
        glUniformMatrix4fv(self.model_matrix_location, 1, GL_FALSE, model_transform)
        glBindVertexArray(engine.mountain_mesh.vao)
        glDrawArrays(GL_LINES, 0, engine.mountain_mesh.vertex_count)

        # grid
        glUniform3fv(self.color_location, 1, engine.palette['Teal'])
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(vec=np.array([-16, -24, 0], dtype=np.float32))
        )
        glUniformMatrix4fv(self.model_matrix_location, 1, GL_FALSE, model_transform)
        glBindVertexArray(engine.grid_mesh.vao)
        glDrawArrays(GL_LINES, 0, engine.grid_mesh.vertex_count)

        # player
        glUniform3fv(self.color_location, 1, engine.palette['Pink'])
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_scale(scale=np.array([0.4, 0.4, 0.4], dtype=np.float32), dtype=np.float32)
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_z_rotation(theta=np.radians(-90), dtype=np.float32)
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_x_rotation(theta=np.radians(scene.player.eulers[0]), dtype=np.float32)
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(vec=scene.player.position, dtype=np.float32)
        )
        glUniformMatrix4fv(self.model_matrix_location, 1, GL_FALSE, model_transform)
        glBindVertexArray(engine.player_mesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.player_mesh.vertex_count)

    def destroy(self):
        glDeleteProgram(self.shader)


class Mesh:
    def __init__(self, filename):
        vertices = self.load_mesh(filename)
        self.vertex_count = len(vertices)
        self.vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    def load_mesh(self, filename):
        v = []

        # final, assembled and packed result
        vertices = []

        # open the obj file and read the data
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                first_space = line.find(" ")
                flag = line[0:first_space]
                if flag == "v":
                    # vertex
                    line = line.replace("v ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag == "f":
                    # face, three or more vertices in v/vt/vn form
                    line = line.replace("f ", "")
                    line = line.replace("\n", "")
                    # get the individual vertices for each line
                    line = line.split(" ")
                    face_vertices = []
                    for vertex in line:
                        # break out into [v,vt,vn],
                        # correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        face_vertices.append(v[position])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i + 1)
                        vertex_order.append(i + 2)
                    for i in vertex_order:
                        for j in face_vertices[i]:
                            vertices.append(j)
                line = f.readline()
        return vertices

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Grid:
    def __init__(self, size):
        vertices = []
        for i in range(size):
            vertices.append(i)
            vertices.append(0)
            vertices.append(0)
            vertices.append(i)
            vertices.append(size - 1)
            vertices.append(0)
        for j in range(size):
            vertices.append(0)
            vertices.append(j)
            vertices.append(0)
            vertices.append(size - 1)
            vertices.append(j)
            vertices.append(0)

        self.vertex_count = len(vertices)
        self.vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

myApp = App(800, 600)
