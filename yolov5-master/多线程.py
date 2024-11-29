import threading
from queue import Queue
from maix import camera, display, image, nn, app

class CameraThread(threading.Thread):
    def __init__(self, model_path, input_queue):
        super().__init__()
        self.detector = nn.YOLOv5(model=model_path)
        self.cam = camera.Camera(self.detector.input_width(), self.detector.input_height(), self.detector.input_format())
        self.input_queue = input_queue

    def run(self):
        while not app.need_exit():
            img = self.cam.read()
            self.input_queue.put((img, self.detector))

def process_frame(input_queue, output_queue):
    while not app.need_exit():
        img, detector = input_queue.get()
        objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
        if objs:
            for obj in objs:
                img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
                msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
                img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)
        output_queue.put(img)

def main():
    input_queue = Queue()
    output_queue = Queue()

    # 创建并启动相机线程
    thread1 = CameraThread("/root/models/zhjt_int8.mud", input_queue)
    thread2 = CameraThread("/root/models/last_int8.mud", input_queue)
    thread1.start()
    thread2.start()

    # 创建并启动处理帧的线程
    processor_thread = threading.Thread(target=process_frame, args=(input_queue, output_queue))
    processor_thread.start()

    # 主循环
    while not app.need_exit():
        img1 = output_queue.get()
        img2 = output_queue.get()
        
        # 合并两张图像
        merged_img = image.Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
        merged_img.paste(img1, (0, 0))
        merged_img.paste(img2, (img1.width, 0))
        
        # 显示合并后的图像
        display.show(merged_img)

    # 结束所有线程
    thread1.join()
    thread2.join()
    processor_thread.join()

if __name__ == "__main__":
    main()