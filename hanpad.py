import sys
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from train_model import *
import copy

class HanWidget(QWidget):
    def __init__(self,parent=None):
        super(HanWidget, self).__init__(parent)
        self.setWindowTitle("绘图例子") 
        self.pix =  QPixmap()  # 实例化一个 QPixmap 对象
        self.lastPoint =  QPoint() # 起始点
        self.endPoint =  QPoint() #终点
        self.strokes = []
        self.initUi()
        self.orders = []
        
    def initUi(self):
        #窗口大小设置为600*500
        self.resize(600, 500)  
        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(512, 512)
        self.pix.fill(Qt.white)

    def drawText(self, pp, pt, text):
        pp.setFont(QFont('Decorative', 18))
        pp.drawText(pt, text)

    # 重绘的复写函数 主要在这里绘制
    def paintEvent(self, event):
        pp = QPainter(self.pix)

        pen = QPen() # 定义笔格式对象
        pen.setWidth(3)  # 设置笔的宽度
        pp.setPen(pen) #将笔格式赋值给 画笔
        self.pix.fill(Qt.white)
        for s, stroke in enumerate(self.strokes):
            pp.setPen(QColor(0, 0, 0)) 
            for i in range(1, len(stroke)):
                beginPos = QPoint(stroke[i-1]['x'], stroke[i-1]['y'])
                endPos = QPoint(stroke[i]['x'], stroke[i]['y'])
                pp.drawLine(beginPos, endPos)
            if len(self.orders) > s:
                pp.setPen(QColor(255, 0, 0)) 
                beginPos = QPoint(self.strokes[s][0]['x'], self.strokes[s][0]['y'])
                self.drawText(pp, beginPos, str(self.orders[s]))

                
        # 根据鼠标指针前后两个位置绘制直线
        # pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)  # 在画布上画出


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
    def mouseReleaseEvent( self, event):
        if event.button() == Qt.LeftButton :
            if len(self.strokes) > 0:
                self.strokes[-1].append({'x':event.pos().x(), 'y':event.pos().y()})
            self.endPoint = event.pos()
            self.update()

    def reset(self):
        self.strokes.clear()
        self.orders.clear()
        self.update()

    def check_order(self, model:HanOrderModel):
        orders = model.get_order(copy.deepcopy(self.strokes))
        self.orders = orders.split(',')
        self.update()

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        self.resize(600, 500)  
        layout = QVBoxLayout()
        
        han_filename = './labels/han.jsonl'
        comp_filename = './labels/comp.jsonl'
        model_filename = './output/{}/{}_model.4.pt'.format('han_sorder_palm_4f60', 'han_sorder_palm_4f60')
        self.hanOrderModel = HanOrderModel(han_filename, comp_filename, model_filename)
        menuBar = self.menuBar()
        file_menu = menuBar.addMenu("文件(&F)")
        act_start = file_menu.addAction(QIcon("./res/start.png"), "启动(&S)")
        act_pause = file_menu.addAction(QIcon("./res/pause.png"), "暂停(&P)")
        act_reset = file_menu.addAction(QIcon("./ress/reset.png"), "复位(&R)") 
        act_exit  = file_menu.addAction(QIcon("./ress/exit.png"), "退出(&X)")

        file_toolBar = self.addToolBar("文件")
        file_toolBar.addAction(act_start)
        file_toolBar.addAction(act_pause)
        file_toolBar.addAction(act_reset)
        file_toolBar.addAction(act_exit)
        
        act_start.triggered.connect(self.on_act_start)
        act_reset.triggered.connect(self.on_act_reset)
        act_exit.triggered.connect(self.on_act_exit)
        act_pause.triggered.connect(self.on_act_pause)

        self.statusBar = self.statusBar()
        self.label_status = QLabel("版本号：1.0")
        self.statusBar.addPermanentWidget(self.label_status)
        self.hanPad = HanWidget()
        layout.addWidget(self.hanPad)
        layout1 = QHBoxLayout()
        layout1.addWidget(QPushButton("笔顺"))
        layout1.addWidget(QPushButton("部件"))
        layout.addLayout(layout1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
    def on_act_start(self):
        self.hanPad.check_order(self.hanOrderModel)
        pass

    def on_act_reset(self):
        self.hanPad.reset()
        pass

    def on_act_pause(self):
        pass

    def on_act_exit(self):
        self.close()

if __name__ == "__main__":  
    app = QApplication(sys.argv) 
    form = MainWindow()
    form.show()
    app.exec()
