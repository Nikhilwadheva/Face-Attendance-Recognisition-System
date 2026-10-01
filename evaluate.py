"""Measure recognition accuracy for the project report.

Run after enrolling >= 5 people (5+ samples each):  python evaluate.py
- Genuine test: hold out one sample per person, train on the rest -> false reject rate (FRR).
- Impostor test: remove one person from training, test their samples -> false accept rate (FAR).
Prints a threshold sweep so you can justify FACE_DISTANCE_THRESHOLD with numbers.
"""
import os, sqlite3, sys
import cv2, numpy as np

db = os.environ.get("DATABASE", "instance/attendance.sqlite3")
rows = sqlite3.connect(db).execute("SELECT employee_id, image_path FROM face_samples").fetchall()
data = {}
for emp, path in rows:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        data.setdefault(emp, []).append(img)
data = {k: v for k, v in data.items() if len(v) >= 3}
if len(data) < 3:
    sys.exit("Enroll at least 3 people with 3+ samples each first.")

def train(ids, skip=None):
    imgs, labels = [], []
    for e in ids:
        for i, im in enumerate(data[e]):
            if skip and skip == (e, i):
                continue
            imgs.append(im); labels.append(e)
    r = cv2.face.LBPHFaceRecognizer_create(); r.train(imgs, np.asarray(labels, np.int32)); return r

genuine, impostor = [], []          # (correct_label, distance)
for e in data:
    for i, im in enumerate(data[e]):
        lab, d = train(data, skip=(e, i)).predict(im); genuine.append((lab == e, d))
    r = train([x for x in data if x != e])
    for im in data[e]:
        impostor.append(r.predict(im)[1])

print(f"{len(data)} people, {sum(map(len, data.values()))} samples")
print("threshold | FRR %  | FAR %")
for t in range(40, 131, 10):
    frr = 100 * sum(not (ok and d <= t) for ok, d in genuine) / len(genuine)
    far = 100 * sum(d <= t for d in impostor) / len(impostor)
    print(f"{t:9d} | {frr:6.1f} | {far:6.1f}")
