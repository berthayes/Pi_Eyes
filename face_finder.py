import cv2
import depthai
import numpy as np
import time
from threading import Thread
from pathlib import Path


class uface(Thread):

    def __init__(self, *args, **kwargs):
        super(uface, self).__init__(*args, **kwargs) # maybe thread?
        try:
            self.pipeline = depthai.Pipeline()
        except:
            print("can't make a pipeline")
        self.cam_rgb = self.pipeline.createColorCamera()
        self.cam_rgb.setPreviewSize(300, 300)
        self.cam_rgb.setInterleaved(False)
        #self.cam_rgb.setImageOrientation(depthai.CameraImageOrientation.ROTATE_180_DEG)
        self.cam_rgb.setImageOrientation(depthai.CameraImageOrientation.HORIZONTAL_MIRROR)

        self.detection_nn = self.pipeline.createMobileNetDetectionNetwork()
        self.detection_nn.setBlobPath(str((Path(__file__).parent / Path('face-detection-retail-0004.blob')).resolve().absolute()))
        self.detection_nn.setConfidenceThreshold(0.5)
        self.cam_rgb.preview.link(self.detection_nn.input)

        self.xout_rgb = self.pipeline.createXLinkOut()
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)

        self.xout_nn = self.pipeline.createXLinkOut()
        self.xout_nn.setStreamName("nn")
        self.detection_nn.out.link(self.xout_nn.input)

        self.cp = ()

        print("init is done")

        


    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it


    def lookit(frame, detections):
        #while True:

        if frame is not None:
            for detection in detections:
                bbox = uface.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                #cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # Preview box is 300 x 300
                # cv2.rectangle uses start_point (X,Y) end_point (X,Y)
                # upper left corner, lower right corner
                # print(bbox[0], bbox[1], bbox[2], bbox[3])
                # 121 45 300 300
                # Upper-left X1, Uppler-Left Y1, lower-right X2, lower-right Y2
                # Original snake_eyes_bonnet returns 0.0 < values < 1.0
                # So divide 300x300 bounding box coords by 300
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                centerx = (x1 + int((x2 - x1)/2)) / 300
                centery = (y1 + int((y2 - y1)/2)) / 300
                center_point = centerx, centery
                if center_point is None:
                    center_point = 0, 0
                #print(center_point)
                return center_point


    def run(self):
        with depthai.Device(self.pipeline) as device:
            q_rgb = device.getOutputQueue("rgb")
            q_nn = device.getOutputQueue("nn")

            frame = None
            detections = []
            

            while True:
                in_rgb = q_rgb.tryGet()
                in_nn = q_nn.tryGet()

                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()

                if in_nn is not None:
                    detections = in_nn.detections

                if detections is not None:
                    self.cp = uface.lookit(frame, detections)
                    #print(self.cp)
                    #return cp
    
    def get_face(self):
        return self.cp


if __name__ == "__main__":
    arf = uface()
    arf.run()

    