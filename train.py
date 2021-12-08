from API import *
from sklearn.preprocessing import LabelEncoder


def train():
    io = IO.getInstance()

    data, labels = io.load_data("dataset/train")
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    distributions = [[] for _ in range(len(le.classes_))]

    for image, label in zip(data, labels):
        gray = convertChannel(image, "BGR2GRAY")
        face_region = face_classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3)

        if len(face_region) > 0:
            x, y, width, height = face_region[0]
            face = gray[y:y+height, x:x+width]
            
            lpbh = LBPH(face)
            distribution = lpbh.getDistribution()
            distributions[label].append(distribution)
    
    distributions = [np.mean(np.array(distribution), axis=0) for distribution in distributions]

    io.save_model("model/distributions", distributions)
    io.save_model("model/labels", le)
        

if __name__ == "__main__":
    train()
