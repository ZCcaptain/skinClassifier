import insightface
import urllib
import urllib.request
import cv2
import numpy as np 

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


img = cv2.imread('/home/zhaochao/workspace/skinClassifier/data/yellow/1.jpg')
cv2.imshow('s', img)
cv2.waitKey(0)
model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id = ctx_id, nms=0.4)

faces = model.get(img)
for idx, face in enumerate(faces):
  # print("Face [%d]:"%idx)
  # print("\tage:%d"%(face.age))
  # gender = 'Male'
  # if face.gender==0:
    # gender = 'Female'
  # print("\tgender:%s"%(gender))
  # print("\tembedding shape:%s"%face.embedding.shape)
  print(face.embedding)
  print(type(face.embedding))
  # print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
  # print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
  print("")