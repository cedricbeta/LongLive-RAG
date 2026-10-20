"""Uncalibrated face-detection/aligned-embedding evidence for synthetic video.

No branch acceptance or real-world identity lookup. Single-subject evidence is
required; a missing/ambiguous detection produces unknown rather than a match.
Thresholds come from the pinned OpenCV Zoo demo, not the observed repair pair.
"""
from pathlib import Path

import cv2
import numpy as np


def diagnostic_decision(scores, complete, threshold=0.363):
    if not complete or not scores or not np.isfinite(scores).all():
        return {'status': 'unknown', 'reason': 'Missing, ambiguous or invalid face evidence'}
    if min(scores) >= threshold:
        return {'status': 'pass', 'reason': 'All observed cross-pair scores exceed the fixed demo threshold'}
    if max(scores) < threshold:
        return {'status': 'fail', 'reason': 'All observed cross-pair scores fall below the fixed demo threshold'}
    return {'status': 'unknown', 'reason': 'Cross-frame evidence straddles the fixed threshold'}


class FaceEvidence:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        cv2.setNumThreads(4)
        self.detector = cv2.FaceDetectorYN.create(str(model_dir / 'face_detection_yunet_2023mar.onnx'),
            '', (320, 320), score_threshold=0.9, nms_threshold=0.3, top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU)
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_dir / 'face_recognition_sface_2021dec.onnx'),
            '', backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU)

    def observe(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError('Cannot read evidence image: ' + str(image_path))
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        faces = np.empty((0, 15), np.float32) if faces is None else faces
        report = {'image_shape': list(image.shape), 'detected_faces': len(faces),
                  'detections_xywh_landmarks_score': faces.tolist(), 'embedding': None}
        if len(faces) != 1:
            return report, None
        crop = self.recognizer.alignCrop(image, faces[0])
        vector = self.recognizer.feature(crop).copy().reshape(-1)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            return report, crop
        report['embedding'] = (vector / np.linalg.norm(vector)).astype(float).tolist()
        return report, crop

    def compare(self, observations_a, observations_b):
        complete = bool(observations_a and observations_b) and all(
            x['detected_faces'] == 1 and x['embedding'] is not None for x in observations_a + observations_b)
        pairs = []
        for i, a in enumerate(observations_a):
            for j, b in enumerate(observations_b):
                if a['embedding'] is not None and b['embedding'] is not None:
                    score = float(np.dot(a['embedding'], b['embedding']))
                    pairs.append({'index_a': i, 'index_b': j, 'cosine_similarity': score})
        return {'diagnostic_decision': diagnostic_decision([p['cosine_similarity'] for p in pairs], complete),
                'all_cross_pairs': pairs, 'evidence_complete': complete,
                'cosine_threshold': 0.363, 'acceptance_policy_calibrated': False}
