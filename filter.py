import cv2 
import dlib
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
MOUTH = list(range(48,61))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
FRONT_FACIAL_CLASSIFIER = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class Santa_beard:
	def beard_size(self, beard):
		self.beard = beard

		beardWidth = dist.euclidean(beard[0], beard[3])
		hull = ConvexHull(beard)
		beardCenter = np.mean(beard[hull.vertices, :], axis = 0)
		beardCenter = beardCenter.astype(int)
		return int(beardWidth), beardCenter

	def place_beard(self, frame, beardCenter, beardSize):
		self.frame = frame
		self.beardCenter = beardCenter
		self.beardSize = beardSize

		beardSize = int(beardSize * 5)
		x1 = int(beardCenter[0,0] - (beardSize/2))
		x2 = int(beardCenter[0,0] + (beardSize/2))
		y1 = int(beardCenter[0,0] - (beardSize/2))
		y2 = int(beardCenter[0,0] + (beardSize/2))

		h, w = frame.shape[:2]

		if x1 < 0:
			x1 = 2
		if y1 < 0:
			y1 = 2
		if x2 > w:
			x2 = w
		if y2 > h:
			y2 = h

		beardOverlayWidth = x2 - x1
		beardOverlayHeight = y2 - y1
 
		beardOverlay = cv2.resize(img_beard, (beardOverlayWidth,beardOverlayHeight), interpolation = cv2.INTER_AREA)
		mask = cv2.resize(orig_mask, (beardOverlayWidth,beardOverlayHeight), interpolation = cv2.INTER_AREA)
		mask_inv = cv2.resize(orig_mask_inv, (beardOverlayWidth,beardOverlayHeight), interpolation = cv2.INTER_AREA)
 
		roi = frame[y1:y2, x1:x2]
 
		roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

		roi_fg = cv2.bitwise_and(beardOverlay,beardOverlay,mask = mask)
		dst = cv2.add(roi_bg,roi_fg)
 
		frame[y1:y2, x1:x2] = dst

class Santa_hat:
	def overlay(self, roi, overlay_image):
		self.roi = roi
		self.overlay_image = overlay_image

		resized_dimensions = (roi.shape[1], roi.shape[0])
		resized_overlay_dst = cv2.resize(
			overlay_image,
			resized_dimensions,
			fx=0,
			fy=0,
			interpolation=cv2.INTER_NEAREST)

		bgr_image = resized_overlay_dst[:, :, :3]
		alpha_mask_1 = resized_overlay_dst[:, :, 3:]
		alpha_mask_2 = cv2.bitwise_not(alpha_mask_1)

		three_chan_alpha_mask_1 = cv2.cvtColor(alpha_mask_1, cv2.COLOR_GRAY2BGR)
		three_chan_alpha_mask_2 = cv2.cvtColor(alpha_mask_2, cv2.COLOR_GRAY2BGR)
		background_roi = (roi * 1 / 255.0) * (three_chan_alpha_mask_2 * 1 / 255.0)
		foreground_roi = (bgr_image * 1 / 255.0) * \
          	(three_chan_alpha_mask_1 * 1 / 255.0)

		return cv2.addWeighted(background_roi, 255.0, foreground_roi, 255.0, 0.0)

	def overlay_img_above_facial_frame(self, facial_frame, x, w, y, h, overlay_t_img):
		santahat = Santa_hat()
		self.facial_frame = facial_frame
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.overlay_t_img = overlay_t_img

		top_of_frame_coord = max(0, y - h)
		rightmost_frame_coord = x + w

		roi_above_face = facial_frame[top_of_frame_coord:y,
                                    x:rightmost_frame_coord]
		overlayed_roi = santahat.overlay(roi_above_face, overlay_t_img)
		facial_frame[top_of_frame_coord:y, x:rightmost_frame_coord] = overlayed_roi

		pass

  
	def get_face_rects_in_frame(self, frame):
		self.frame = frame	

		return FRONT_FACIAL_CLASSIFIER.detectMultiScale(
			frame, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

class Landmark:
	def PIL2array(self, img):
		self.img = img
		return np.array(img.getdata(),
			np.uint8).reshape(img.size[1], img.size[0], 3)
	def up_sample(self, landmark_list , sample_size=4):
		self.landmark_list = landmark_list
		for face_landmark in landmark_list:
			if len(face_landmark) > 1:
				for key in face_landmark.keys():
					face_landmark[key] = [(w[0]*sample_size , w[1]*sample_size) for w in face_landmark[key]]
		return landmark_list

class FaceLandMarkDetection:
	def predict(self , frame):
		self.frame = frame
		face_landmarks = face_recognition.face_landmarks(frame)
		if down_sampling:
			landmark = Landmark()
			self.face_landmarks = landmark.up_sample(face_landmarks)
		else:
			self.face_landmarks = face_landmarks
  
  
	def plot(self , frame):
		pil = Landmark()
		self.frame = frame
		pil_image = Image.fromarray(frame)
		
		for face_landmarks in self.face_landmarks:
			if len(face_landmarks) > 1:
				d = ImageDraw.Draw(pil_image, 'RGBA')
				d.polygon(face_landmarks['left_eyebrow'], fill=(255, 255, 255, 128))
				d.polygon(face_landmarks['right_eyebrow'], fill=(255, 255, 255,128))
		#		d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255, 150), width=5)
		#		d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255, 150), width=5)
		return pil.PIL2array(pil_image)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
img_beard = cv2.imread("santa_beard.png",-1)
img_hat = cv2.imread("santa_hat1.png",-1)
orig_mask = img_beard[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
img_beard = img_beard[:,:,0:3]
origBeardHeight, origBeardWidth = img_beard.shape[:2]
face_landmark_detection = FaceLandMarkDetection()
santa_beard = Santa_beard()
santa_hat = Santa_hat()
down_sampling = True

while True:
	ret, frame = cap.read()
	if ret:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
		rects = detector(gray, 0)

		for rect in rects:
			x = rect.left()
			y = rect.top()
			x1 = rect.right()
			y1 = rect.bottom()
	
			landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
			mouth = landmarks[MOUTH]
			BeardSize, BeardCenter = santa_beard.beard_size(mouth)
			santa_beard.place_beard(frame, BeardCenter, BeardSize)

		if down_sampling:
			small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
			face_landmark_detection.predict(small_frame)
		else:
			face_landmark_detection.predict(frame)
			frame = face_landmark_detection.plot(frame)


		face_rects = santa_hat.get_face_rects_in_frame(gray)
		for (index, (x, y, w, h)) in enumerate(face_rects):
			santa_hat.overlay_img_above_facial_frame(frame, x, w, y, h, img_hat)


		cv2.imshow("Faces with Overlay", frame)

	ch = 0xFF & cv2.waitKey(1)
	if ch == ord('q'):
		break 

cv2.destroyAllWindows()
	



