import sys
import platform

def main():
    print('Project smoke test')
    print('Python:', sys.version.splitlines()[0])
    print('Platform:', platform.platform())

    try:
        import cv2
        print('cv2:', cv2.__version__)
    except Exception as e:
        print('cv2 import failed:', e)

    try:
        import numpy as np
        print('numpy:', np.__version__)
    except Exception as e:
        print('numpy import failed:', e)

    try:
        import torch
        print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())
    except Exception as e:
        print('torch import failed:', e)

    try:
        import ultralytics
        print('ultralytics:', ultralytics.__version__)
    except Exception as e:
        print('ultralytics import failed:', e)

if __name__ == '__main__':
    main()
