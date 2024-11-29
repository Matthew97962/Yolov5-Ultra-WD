import os
import xml.etree.ElementTree as ET

# 配置
#classes = ['LD','HD','zhixing','P','zuo','you']  # 替换为您的类别名称
classes = ['LD','HD','zuo','you','zhixing','P']
#classes = ['start', 'noentry', 'forwoard', 'backward', 'right', 'left', 'honk', 'turnaround', 'park', 'end']  # 替换为您的类别名称
#input_dir = 'I:/yolov5-master/2023jtbzhi/xml/'          # xml 文件所在文件夹路径
input_dir = 'E:/jtbz/'
#output_dir = 'I:/yolov5-master/2023jtbzhi/labels/'  # 输出 txt 文件夹路径
output_dir = 'E:/jtbz/'  # 输出 txt 文件夹路径

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    txt_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
    with open(txt_file, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for xml_file in os.listdir(input_dir):
    if xml_file.endswith('.xml'):
        convert_annotation(os.path.join(input_dir, xml_file))
