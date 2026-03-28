import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

# Mock settings
os.environ["APP_ENV"] = "test"
os.environ["UPLOAD_DIR"] = "uploads"
os.environ["ALLOWED_ORIGINS"] = "*"

import numpy as np
from app.services.inference import inference_service
from app.schemas.schemas import StrokeType

def test():
    # Test with "normal" filename
    res1 = inference_service.run_classification(np.zeros((256,256)), "test1", "normal_scan.jpg")
    print(f"Normal Filename -> {res1.stroke_type} ({res1.severity}) - Conf: {res1.confidence}")

    # Test with "stroke" filename
    res2 = inference_service.run_classification(np.zeros((256,256)), "test2", "ischemic_scan.jpg")
    print(f"Ischemic Filename -> {res2.stroke_type} ({res2.severity}) - Conf: {res2.confidence}")

    # Test with neutral filename
    res3 = [inference_service.run_classification(np.zeros((256,256)), f"test{i}", "scan.jpg") for i in range(10)]
    for i, r in enumerate(res3):
        print(f"Neutral {i} -> {r.stroke_type} ({r.severity}) - Conf: {r.confidence}")

if __name__ == "__main__":
    test()
