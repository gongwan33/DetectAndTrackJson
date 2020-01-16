import json
import numpy as np
import _thread
import time

class JSONOutput:
    def __init__(self, fname, conf):
        ofname = ''.join(fname.split(".")[:-1])

        if len(ofname) < 1:
            ofname = fname + '-res.json'
        else:
            ofname = ofname + '.json'

        print("Open %s to write json data"%ofname)
        self.f = open(ofname, 'w+')

        if self.f is not None:
            self.f.write("[")

        self.conf = conf
        self.anno_count = 0
        self.gpu_percent = 0
        self.gpu_memory = 0
        self.gpu_thread = True
        self.thread_lock = _thread.allocate_lock()

        #try:
        #    _thread.start_new_thread(self.get_gpu_info, ())
        #    print("GPUINFO Thread Started.")
        #except Exception as e:
        #    print("Unable to start GPUINFO Thread")
        #    print(e)

        return

    def get_gpu_info(self):
        print("GPUINFO: Start loop")
        while self.gpu_thread:
            self.thread_lock.acquire()
            self.gpu_percent, self.gpu_memory = GPUInfo.gpu_usage()
            self.thread_lock.release()
            time.sleep(2)

        print("GPUINFO: End loop")
        return

    def write(self, boxes, keyps, idx):
        if boxes is None or keyps is None or len(boxes) <= 0 or len(keyps) <= 0:
            return

        conf = self.conf

        boxes = np.array(boxes[1])
        scores = boxes[:, 4]
        filtered_scores = scores[scores >= conf]

        for i, points in enumerate(keyps[1]):
            points = np.array(points)
            points = np.delete(points, 2, 0).transpose()
            score = scores[i]

            if score < conf:
                continue

            if len(points) <= 0:
                continue

            points_ary = points.flatten().tolist()
            scores_ary = scores[i].astype("float").round(5)
            instance_num = len(filtered_scores)

            self.thread_lock.acquire()
            key_annotation = {
                "image_id": idx,
                "category_id": 1,
                "keypoints": points_ary,
                "score": scores_ary,
                "instance_num": instance_num,
                "gpu_percentage": self.gpu_percent,
                "gpu_memory": self.gpu_memory
            }
            self.thread_lock.release()

            if self.f is not None:
                if self.anno_count > 0:
                    self.f.write(",\n")

                json.dump(key_annotation, self.f)
                self.anno_count += 1

        return

    def release(self):
        if self.f is not None:
            self.f.write("]")
            self.f.close()
            self.gpu_thread = False
        return

