import cv2
from src.drawer import Drawer
from src.threads import CameraThread, DetectionThread

def main() -> None:
    cam_t = CameraThread()
    det_t = DetectionThread(cam_t)
    drawer = Drawer()

    cam_t.start()
    det_t.start()

    while True:
        frame = cam_t.get_frame()
        
        if frame is not None:
            results, _ = det_t.get_data()

            if results:
                frame = drawer.draw(frame, results)
            
            cam_t.camera.show_frame(frame, "BSL")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam_t.stop()
    det_t.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()