import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
import csv

def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(path, img_size=(224,224)):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = tf.expand_dims(img, 0)
    return img

def predict_folder(model, class_names, folder, out_csv='predictions.csv'):
    folder = Path(folder)
    images = sorted([p for p in folder.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']])
    rows = []
    for p in images:
        img = preprocess_image(p)
        preds = model.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))
        rows.append([str(p), class_names[idx], float(preds[idx])])
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path','predicted_class','score'])
        writer.writerows(rows)
    return out_csv

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python infer_local.py model_path class_names.txt images_folder')
        sys.exit(1)
    model = load_model(sys.argv[1])
    class_names = [l.strip() for l in open(sys.argv[2]).read().splitlines()]
    out = predict_folder(model, class_names, sys.argv[3])
    print('predictions saved to', out)
