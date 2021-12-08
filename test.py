import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.colors import NoNorm
from API import *


def show_images(images, titles, figsize=(8,6), row=1, col=1):
  plt.figure(figsize=figsize)
  for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(row, col, i+1)
    if image.ndim == 2:
      plt.imshow(image, cmap="gray", norm=NoNorm())
    else:
      plt.imshow(image)
    plt.title(title)
  
  plt.show()


def test():
    io = IO.getInstance()

    images, names = [], []
    
    model = io.load_model("model/distributions")
    le = io.load_model("model/labels")
    data, labels = io.load_data("dataset/test")

    testY, predict = [], []

    for image, identity in zip(data, labels):
        gray = convertChannel(image, "BGR2GRAY")
        # face detection
        face_region = face_classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3)
        
        if len(face_region) > 0:
          x, y, width, height = face_region[0]
          face = gray[y:y+height, x:x+width]

          lpbh = LBPH(face)
          distribution = lpbh.getDistribution()
          
          error = [np.linalg.norm(distribution-pattern) for pattern in model]
          confidence = min(error)
          regconization = np.argmin(error)          

          image = convertChannel(image, "BGR2RGB")
          name = le.classes_[regconization]

          testY.append(identity)
          predict.append(name)

          cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
          cv2.putText(image, name + f' {round(confidence, 2)}', (x+5, y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
          name += " - True" if name == identity else " - False"

        images.append(image)
        names.append(name.replace("_", " "))
  
    testY = le.transform(testY)
    predict = le.transform(predict)
    print(classification_report(testY, predict, target_names=le.classes_))

    show_images(images, names, (24, 8), 2, len(images) // 2)


if __name__ == "__main__":
    test()