from maix import camera, display, image, nn, app,uart,time
device = "/dev/ttyS0"
serial = uart.UART(device, 9600)
detector01 = nn.YOLOv5(model="/root/models/zhjt_int8.mud")
# detector = nn.YOLOv8(model="/root/models/yolov8n.mud")
lb=''
sb=''
cam = camera.Camera(detector01.input_width(), detector01.input_height(), detector01.input_format())

dis = display.Display()

while not app.need_exit():
    sb = serial.read()
    img = cam.read()    
    objs = detector01.detect(img, conf_th = 0.9, iou_th = 0.45)    
    if objs and sb.decode() == 'daole':
        for obj in objs:
                img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
                msg = f'{detector01.labels[obj.class_id]}: {obj.score:.2f}'
                lb = detector01.labels[obj.class_id] 
                print(lb)
                img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        sb=''
        if lb == 'you':
            print('okyou')
            serial.write_str("you")
            lb=''
            time.sleep_ms(100)
            
        elif lb=='zuo':
            print('okzuo')
            serial.write_str("zuo")
            lb=''
            time.sleep_ms(100)
        elif lb=='p':
            print('okp')
            serial.write_str("p")
            lb=''
            time.sleep_ms(100)
        elif lb=='LD':
             print('okLD')
             serial.write_str("LD")
             lb=''
             time.sleep_ms(100)
        elif lb=='HD':
             print('okHD')
             serial.write_str("HD")
             lb=''
             time.sleep_ms(100)
        elif lb=='zhixing':
             print('okzhixing')
             serial.write_str("zhixing")
             lb=''
             time.sleep_ms(100)
        
    
        dis.show(img)
    




        

   
