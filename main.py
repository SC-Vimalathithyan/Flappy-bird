import cv2
import numpy as np
import dlib
import pygame
import random
import imutils
import math
from imutils.video import VideoStream

SHAPE_DETECTOR = "shape_predictor_68_face_landmarks.dat" 
MAR_THRESHOLD = 1.5
MAR_CONSECUTIVE_FRAMES = 3  

(MOUTH_LM_INDEX_START, MOUTH_LM_INDEX_END) = (48, 68)

FONT = cv2.FONT_HERSHEY_SIMPLEX
PREVIEW_TEXT_COLOUR = (0, 255, 255)
PREVIEW_MOUTH_COLOUR = (0, 255, 0)
GAME_SCORE_TEXT_COLOUR = (255, 255, 255)
GAME_BG_COLOUR = (212, 182, 115)

JUMP_HEIGHT = 17
JUMP_GRAVITY = 5
JUMP_SPEED = 10
BIRD_Y_DEFAULT = 150
BIRD_X = 70
WALL_GAP = 220

def landmark_shape_to_np(lm_shape):
	coordinates = np.zeros((lm_shape.num_parts, 2), dtype="int")
	for i in range(0, lm_shape.num_parts):
		coordinates[i] = (lm_shape.part(i).x, lm_shape.part(i).y)

	return coordinates
def distance(point1, point2):
	dx = point1[0] - point2[0]
	dy = point1[1] - point2[1]
	return math.sqrt(dx * dx + dy * dy)
def mouth_aspect_ratio(mouth_landmarks):

	vertical1_d = distance(mouth_landmarks[2], mouth_landmarks[10]) 
	vertical2_d = distance(mouth_landmarks[4], mouth_landmarks[8]) 
	horizontal_d = distance(mouth_landmarks[0], mouth_landmarks[6]) 

	mar = (vertical1_d + vertical2_d) / horizontal_d
	return mar

def draw_text(image, text, origin, colour):
	cv2.putText(image, text, origin, FONT, 0.7, colour, 2)

class FlappyBird:
	def __init__(self):
		self.screen = pygame.display.set_mode((500, 800))
		self.bird = pygame.Rect(65, 50, 50, 50)
		self.background = pygame.image.load("images/bg.png").convert()
		self.birdSprites = [pygame.image.load("images/1.png").convert_alpha(),
							pygame.image.load("images/2.png").convert_alpha(),
							pygame.image.load("images/dead.png")]
		self.wallUp = pygame.image.load("images/bottom.png").convert_alpha()
		self.wallDown = pygame.image.load("images/top.png").convert_alpha()
		self.gap = WALL_GAP
		self.wallx = 400
		self.birdY = BIRD_Y_DEFAULT
		self.jump = 0
		self.jumpSpeed = JUMP_SPEED
		self.gravity = JUMP_GRAVITY
		self.dead = False
		self.sprite = 0
		self.counter = 0
		self.offset = random.randint(-110, 110)

	def updateWalls(self):
		self.wallx -= 2
		if self.wallx < -80:
			self.wallx = 400
			self.counter += 1
			self.offset = random.randint(-110, 110)

	def birdUpdate(self):
		if self.jump:
			self.jumpSpeed -= 1
			self.birdY -= self.jumpSpeed
			self.jump -= 1
		else:
			self.birdY += self.gravity
			self.gravity += 0.15
		self.bird[1] = int(self.birdY)
		wall_up_react = pygame.Rect(self.wallx, 360 + self.gap - self.offset + 10, self.wallUp.get_width() - 10,
									self.wallUp.get_height())
		wall_down_react = pygame.Rect(self.wallx, 0 - self.gap - self.offset - 10, self.wallDown.get_width() - 10,
									  self.wallDown.get_height())
		if wall_up_react.colliderect(self.bird):
			self.dead = True
		if wall_down_react.colliderect(self.bird):
			self.dead = True
		if not 0 < self.bird[1] < 720:
			self.bird[1] = BIRD_Y_DEFAULT
			self.birdY = BIRD_Y_DEFAULT
			self.dead = False
			self.counter = 0
			self.wallx = 400
			self.offset = random.randint(-110, 110)
			self.gravity = JUMP_GRAVITY

	def run(self):
		counter_consec_frame = 0 
		total_open = 0  

		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(SHAPE_DETECTOR)

		vs = VideoStream(src=0).start()

		clock = pygame.time.Clock()
		pygame.font.init()
		game_font = pygame.font.SysFont("Arial", 50)

		game_over = False
		while not game_over:
			frame = vs.read()
			frame = imutils.resize(frame, width=450)

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = detector(gray, 0)
			count_face = len(rects)
			if count_face == 0:
				draw_text(frame, "No face detected".format(len(rects)), (10, 30), PREVIEW_TEXT_COLOUR)
			elif count_face > 1:
				draw_text(frame, "Only support 1 face. Detected {0} faces".format(count_face), (10, 30),
						  PREVIEW_TEXT_COLOUR)
			else:
				rect = rects[0]

				shape = predictor(gray, rect)
				shape = landmark_shape_to_np(shape)


				mouth = shape[MOUTH_LM_INDEX_START:MOUTH_LM_INDEX_END]

				hull = cv2.convexHull(mouth)

				cv2.drawContours(frame, [hull], -1, PREVIEW_MOUTH_COLOUR, 1)

				mar = mouth_aspect_ratio(mouth)
				draw_text(frame, "Mouth Aspect Ratio: {:.2f}".format(mar), (10, 30), PREVIEW_TEXT_COLOUR)
				if mar < MAR_THRESHOLD:
					counter_consec_frame += 1
				else:
					draw_text(frame, "Mouth OPEN", (10, 60), PREVIEW_TEXT_COLOUR)
					if counter_consec_frame >= MAR_CONSECUTIVE_FRAMES and not self.dead:
						total_open += 1 
						self.jump = JUMP_HEIGHT
						self.gravity = JUMP_GRAVITY
						self.jumpSpeed = JUMP_SPEED
					counter_consec_frame = 0

			cv2.imshow("Flappy Bird", frame)
			clock.tick(60)

			self.screen.fill(GAME_BG_COLOUR)
			self.screen.blit(self.background, (0, 0))
			self.screen.blit(self.wallUp, (self.wallx, 360 + self.gap - self.offset))
			self.screen.blit(self.wallDown, (self.wallx, 0 - self.gap - self.offset))

			self.screen.blit(game_font.render(str(self.counter), -1, GAME_SCORE_TEXT_COLOUR), (250, 50))

			if self.dead:
				self.sprite = 2
			elif self.jump:
				self.sprite = 1
			self.screen.blit(self.birdSprites[self.sprite], (BIRD_X, int(self.birdY)))

			if not self.dead:
				self.sprite = 0

			self.updateWalls()

			self.birdUpdate()

			pygame.display.update()
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				game_over = True

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					game_over = True
		cv2.destroyAllWindows()
		vs.stop()


if __name__ == "__main__":
	FlappyBird().run()
