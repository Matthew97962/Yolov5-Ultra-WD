from maix import camera, display, image, nn, app

detector01 = nn.YOLOv5(model="/root/models/zhjt_int8.mud")
detector02 = nn.YOLOv5(model="/root/models/last_int8.mud")
# detector = nn.YOLOv8(model="/root/models/yolov8n.mud")

cam = camera.Camera(detector01.input_width(), detector01.input_height(), detector01.input_format())
cam02 = camera.Camera(detector02.input_width(), detector02.input_height(), detector02.input_format())
dis = display.Display()

while not app.need_exit():
    img = cam.read()    
    objs = detector01.detect(img, conf_th = 0.5, iou_th = 0.45)    
    if objs:
        for obj in objs:
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
            msg = f'{detector01.labels[obj.class_id]}: {obj.score:.2f}'
            img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        dis.show(img)

    img02 = cam02.read()
    objs02 = detector02.detect(img02, conf_th = 0.5, iou_th = 0.45)
    if objs02:
        for obj02 in objs02:
            img02.draw_rect(obj02.x, obj02.y, obj02.w, obj02.h, color = image.COLOR_RED)
            msg = f'{detector02.labels[obj02.class_id]}: {obj02.score:.2f}'
            img02.draw_string(obj02.x, obj02.y, msg, color = image.COLOR_RED)
         
        dis.show(img02)
