import sys
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from train_model import *
import copy

class HanWidget(QWidget):
    def __init__(self,parent=None):
        super(HanWidget, self).__init__(parent)
        # self.setWindowTitle("绘图例子") 
        self.pix =  QPixmap()  # 实例化一个 QPixmap 对象
        self.lastPoint =  QPoint() # 起始点
        self.endPoint =  QPoint() #终点
        self.strokes = []
        self.comps = []
        self.compRange = []
        self.initUi()
        self.orders = []
        
    def initUi(self):
        #窗口大小设置为600*500
        self.resize(500, 500)  
        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(1000, 1000)
        self.pix.fill(Qt.white)

    def drawText(self, pp, pt, text):
        pp.setFont(QFont('Decorative', 18))
        pp.drawText(pt, text)

    # 重绘的复写函数 主要在这里绘制
    def paintEvent(self, event):
        pp = QPainter(self.pix)
        pp.setRenderHint(QPainter.Antialiasing)
        pen = QPen() # 定义笔格式对象
        pen.setWidth(3)  # 设置笔的宽度
        pp.setPen(pen) #将笔格式赋值给 画笔
        self.pix.fill(Qt.white)
        if len(self.compRange) > 0:
            for s, comp in enumerate(self.compRange):
                for c in range(comp['begin'], comp['end']):
                    pp.setPen(QColor(s * 255, 0, 0)) 
                    if c < len(self.strokes):
                        stroke = self.strokes[c]
                        for i in range(1, len(stroke)):
                            beginPos = QPoint(stroke[i-1]['x'], stroke[i-1]['y'])
                            endPos = QPoint(stroke[i]['x'], stroke[i]['y'])
                            pp.drawLine(beginPos, endPos)
        else:
            for s, stroke in enumerate(self.strokes):
                if len(self.orders) > s:
                    # pp.setPen(QColor(0, 0, 0))
                    pp.setPen(QColor(255, 0, 0))                     
                    beginPos = QPoint(stroke[0]['x'], stroke[0]['y'])
                    self.drawText(pp, beginPos, str(self.orders[s]))
                else:
                    pp.setPen(QColor(0, 0, 0)) 
                for i in range(1, len(stroke)):
                    beginPos = QPoint(stroke[i-1]['x'], stroke[i-1]['y'])
                    endPos = QPoint(stroke[i]['x'], stroke[i]['y'])
                    pp.drawLine(beginPos, endPos)

        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)  # 在画布上画出
        painter.end()


   #鼠标按压事件
    def mousePressEvent(self, event) :   
        # 鼠标左键按下  
        if event.button() == Qt.LeftButton :
            self.strokes.append([{'x':event.pos().x(), 'y':event.pos().y()}])
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    # 鼠标移动事件
    def mouseMoveEvent(self, event):    
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton :
            self.endPoint = event.pos()
            if len(self.strokes) > 0:
                self.strokes[-1].append({'x':event.pos().x(), 'y':event.pos().y()})
            #进行重新绘制
            self.update()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton :
            if len(self.strokes) > 0:
                self.strokes[-1].append({'x':event.pos().x(), 'y':event.pos().y()})
            self.endPoint = event.pos()
            self.update()

    def set_strokes(self, strokes):
        self.strokes.clear()
        for each in strokes:
            self.strokes.append([])
            for point in each:
                self.strokes[-1].append({'x': point['x'], 'y':point['y']})

    def set_points(self, points):
        index = 0
        self.strokes.clear()
        self.strokes.append([])
        for each in points:
            if (each[0] == index):
                self.strokes[index].append({'x': each[2], 'y':each[3]})
            else:
                index += 1
                self.strokes.append([])


    def reset(self):
        self.strokes.clear()
        self.orders.clear()
        self.comps.clear()
        self.compRange.clear()

        self.update()

    def check_stroke(self, model:HanStrokeModel):
        self.compRange.clear()
        orders = model.get_order(copy.deepcopy(self.strokes))
        
        self.orders = orders.split(',')

        self.update()

    def check_order(self, model:HanOrderModel):
        self.compRange.clear()
        orders = model.get_order(copy.deepcopy(self.strokes))

        self.orders = orders.split(',')
        strokes = []
        for i in range(len(self.orders)):
            for j, order in enumerate(self.orders):
                if int(order)  == i + 1:
                    strokes.append(self.strokes[j])
        self.strokes = strokes
        self.update()

    def check_comp(self, model:HanCompModel):
        orders = model.get_order(copy.deepcopy(self.strokes))
        self.comps = orders.split(',')
        self.compRange = model.getLabelStrokeRange(self.comps)
        self.update()

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.hw_index =  0
        self.setWindowTitle("汉字笔顺识别")
        self.resize(600, 500)  
        layout = QVBoxLayout()
        self.hanPad = HanWidget()
        han_filename = './labels/han.jsonl'
        comp_filename = './labels/comp.jsonl'

        self.json_filename = './output/palm_4f60_256x256_comp_test.jsonl'
        model_filename = './output/han_sorder_palm_4f60_model.20.pt'
        self.hw_data = []
        f = open(self.json_filename, 'r', encoding='utf_8')
        for each in f:
            jdata = json.loads(each)
            self.hw_data.append(jdata)
        print(len(self.hw_data))
        self.hanPad.set_strokes(self.hw_data[0]['strokes'])
        
        self.hanOrderModel = HanOrderModel(han_filename, comp_filename, model_filename)

        model_filename = './output/han_stroke_casia_model.39.pt'
        self.hanStrokeModel = HanStrokeModel(han_filename, comp_filename, model_filename)

        model_filename = './output/{}/{}_model.10.pt'.format('han_comp_casia', 'han_comp_casia')
        self.hanCompModel = HanCompModel(han_filename, comp_filename, model_filename)

        menuBar = self.menuBar()
        file_menu = menuBar.addMenu("文件(&F)")
        act_start = file_menu.addAction(QIcon("./res/start.png"), "笔顺(&S)")
        act_pause = file_menu.addAction(QIcon("./res/pause.png"), "部件(&R)")
        act_reset = file_menu.addAction(QIcon("./res/reset.png"), "复位(&T)") 
        act_prev = file_menu.addAction(QIcon("./res/prev.png"), "上一个(&P)") 
        act_next = file_menu.addAction(QIcon("./res/next.png"), "下一个(&N)") 
        act_exit  = file_menu.addAction(QIcon("./res/exit.png"), "退出(&X)")

        file_toolBar = self.addToolBar("文件")
        file_toolBar.addAction(act_start)
        file_toolBar.addAction(act_pause)
        file_toolBar.addAction(act_reset)
        file_toolBar.addAction(act_prev)
        file_toolBar.addAction(act_next)
        file_toolBar.addAction(act_exit)
        
        act_start.triggered.connect(self.on_act_start)
        act_reset.triggered.connect(self.on_act_reset)
        act_exit.triggered.connect(self.on_act_exit)
        act_pause.triggered.connect(self.on_act_pause)
        act_next.triggered.connect(self.on_act_next)
        act_prev.triggered.connect(self.on_act_prev)

        self.statusBar = self.statusBar()
        self.label_status = QLabel("版本号：1.0")
        self.statusBar.addPermanentWidget(self.label_status)

        layout.addWidget(self.hanPad)
        # layout1 = QHBoxLayout()
        # layout1.addWidget(QPushButton("笔顺"))
        # layout1.addWidget(QPushButton("部件"))
        # layout.addLayout(layout1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
    def on_act_start(self):
        self.hanPad.check_stroke(self.hanStrokeModel)
        self.hanPad.check_order(self.hanOrderModel)
        
        pass

    def on_act_reset(self):
        self.hanPad.reset()
 
        file_path, _ = QFileDialog.getOpenFileName(None, "选择文件")
        
        if file_path:
            self.json_filename = file_path
            
            self.hw_data = []
            f = open(self.json_filename, 'r', encoding='utf_8')
            for each in f:
                jdata = json.loads(each)
                self.hw_data.append(jdata)
            print(len(self.hw_data))
            self.hanPad.set_strokes(self.hw_data[0]['strokes'])
            

    def on_act_pause(self):
        self.hanPad.check_comp(self.hanCompModel)
        pass

    def on_act_next(self):
        if self.hw_index < len(self.hw_data) - 1:
            self.hw_index += 1
        self.hanPad.set_strokes(self.hw_data[self.hw_index]['strokes'])
        self.on_act_pause()
        self.hanPad.update()
        pass

    def on_act_prev(self):
        if (self.hw_index > 0):
            self.hw_index -= 1
        self.hanPad.set_strokes(self.hw_data[self.hw_index]['strokes'])
        self.on_act_pause()
        self.hanPad.update()
        pass

    def on_act_exit(self):
        self.close()

if __name__ == "__main__":  
    app = QApplication(sys.argv) 
    form = MainWindow()
    form.show()
    app.exec()
