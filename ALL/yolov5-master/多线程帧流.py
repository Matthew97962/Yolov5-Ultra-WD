import threading
from queue import Queue
from maix import camera, display, image, nn, app

# 全局帧缓冲队列
frame_buffer = Queue(maxsize=10)

# 输出队列
output_queue1 = Queue()
output_queue2 = Queue()

class FrameCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        #self.cam = camera.Camera(width=224, height=224, format="rgb")
        self.cam = camera.Camera(width=224, height=224, format=image.Format.FMT_RGB888)

    def run(self):
        while not app.need_exit():
            img = self.cam.read()
            try:
                frame_buffer.put(img, block=False)
            except Queue.Full:
                pass  # 如果队列满了，就跳过这一帧

class CameraThread(threading.Thread):
    def __init__(self, model_path, output_queue):
        super().__init__()
        self.detector = nn.YOLOv5(model=model_path)
        self.output_queue = output_queue

    def run(self):
        while not app.need_exit():
            try:
                img = frame_buffer.get(block=True, timeout=1)
                objs = self.detector.detect(img, conf_th=0.5, iou_th=0.45)
                if objs:
                    for obj in objs:
                        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
                        msg = f'{self.detector.labels[obj.class_id]}: {obj.score:.2f}'
                        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)
                self.output_queue.put(img)
            except Queue.Empty:
                continue

def main():
    # 创建并启动采集线程
    capture_thread = FrameCaptureThread()
    capture_thread.start()

    # 创建并启动处理线程
    thread1 = CameraThread("/root/models/zhjt_int8.mud", output_queue1)
    thread2 = CameraThread("/root/models/last_int8.mud", output_queue2)
    thread1.start()
    thread2.start()

    # 主循环
    while not app.need_exit():
        img1 = output_queue1.get()
        img2 = output_queue2.get()
        
        # 合并两张图像
        #merged_img = image.Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
        #merged_img.paste(img1, (0, 0))
        #erged_img.paste(img2, (img1.width, 0))

        # 合并两张图像
        merged_img = image.Image(width=(img1.width() + img2.width()), 
                                height=max(img1.height(), img2.height()))
        merged_img.draw_image(0, 0, img1)
        merged_img.draw_image(img1.width(), 0, img2)
       
        
        # 显示合并后的图像
        display.Display().show(merged_img)

    # 结束所有线程
    capture_thread.join()
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()