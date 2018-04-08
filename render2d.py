import os
import sys
import math
import time

import numpy as np
import pyglet
from pyglet.gl import *

RAD2DEG = 180.0 / math.pi

class Vertex:
	def __init__(self, color, coord):
		self.color = color
		self.coord = coord

class Geometry:
	def __init__(self):
		self.translation = [0.0, 0.0, 0.0]
		self.rotation_rad = 0.0
		self.prim_type = GL_TRIANGLES
		self.vertices = []
		# Only for line primitives
		self.line_width = 1.0

	def add_vertex(self, color, coord):
		vtx = Vertex(color, coord)
		self.vertices.append(vtx)
		return vtx

	def render(self):
		glPushMatrix()
		glTranslatef(
			self.translation[0],
			self.translation[1],
			self.translation[2]
			)
		glRotatef(
			RAD2DEG * self.rotation_rad,
			0.0,
			0.0,
			1.0
			)
		if self.line_width != 1.0:
			glLineWidth(self.line_width)
		glBegin(self.prim_type)
		for vtx in self.vertices:
			glColor3f(
				vtx.color[0],
				vtx.color[1],
				vtx.color[2]
				)
			glVertex3f(
				vtx.coord[0],
				vtx.coord[1],
				vtx.coord[2]
				)
		glEnd()
		if self.line_width != 1.0:
			glLineWidth(1.0)
		glPopMatrix()

class GeomCircle(Geometry):
	def __init__(self, center, radius, color, is_filled=False, det=24, width=1.0):
		Geometry.__init__(self)
		assert det > 2, "Circle det should be > 2"
		self.prim_type = GL_TRIANGLES if is_filled else GL_LINES
		self.line_width = 1.0 if is_filled else width
		prev_vtx = None
		for idx in range(det):
			coord = [0.0, 0.0, 0.0]
			ang = idx / (det-1) * 2.0 * math.pi
			coord[0] = center[0] + radius * math.cos(ang)
			coord[1] = center[1] + radius * math.sin(ang)
			coord[2] = center[2]
			if prev_vtx:
				if is_filled:
					self.add_vertex(color, center)
				self.vertices.append(prev_vtx)
				prev_vtx = self.add_vertex(color, coord)
			else:
				prev_vtx = Vertex(color, coord)

class GeomLine(Geometry):
	def __init__(self, c0, p0, c1, p1, width = 1.0):
		Geometry.__init__(self)
		self.prim_type = GL_LINES

		self.line_width = width
		self.add_vertex(c0, p0)
		self.add_vertex(c1, p1)

class Renderer:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.aspect = width / height
		self.window = pyglet.window.Window(width=width, height=height, display=None)
		self.window.on_close = self.on_close
		self.is_open = True

		self.geoms = []

		self.clear_color = [1.0, 1.0, 1.0, 1.0]

	def on_close(self):
		self.is_open = False
		#sys.exit(0)

	def reset_geoms(self):
		del (self.geoms[:])

	def render(self):
		if not self.is_open:
			return

		glClearColor(
			self.clear_color[0],
			self.clear_color[1],
			self.clear_color[2],
			self.clear_color[3]
			)

		self.window.clear()
		self.window.switch_to()
		self.window.dispatch_events()

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-self.aspect, self.aspect, -1.0, 1.0, -1.0, 1.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		for geom in self.geoms:
			geom.render()

		self.window.flip()

if __name__ == '__main__':
	test_renderer = Renderer(800, 600)
	geom = Geometry()
	geom.prim_type = GL_LINES
	geom.add_vertex((1.0, 0.0, 0.0), (-0.5, -0.5, 0.0))
	geom.add_vertex((1.0, 0.0, 0.0), ( 0.5,  0.5, 0.0))
	geom.add_vertex((1.0, 0.0, 0.0), (-0.5,  0.5, 0.0))
	geom.add_vertex((1.0, 0.0, 0.0), ( 0.5, -0.5, 0.0))
	test_renderer.geoms.append(geom)

	circle = GeomCircle(
		center=[0.0, 0.0, 0.0],
		radius=0.25,
		color=[0.0, 0.0, 1.0],
		det=24,
		is_filled=False
		)
	test_renderer.geoms.append(circle)

	elapsed_time = 0.0

	prev_time = time.time()
	while True:
		cur_time = time.time()
		dt = cur_time - prev_time
		prev_time = cur_time
		elapsed_time += dt
		geom.rotation_rad += 0.1 * dt
		test_renderer.render()