import AES
import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QMessageBox

from PyQt5.uic import loadUi  # 导入 loadUi 函数
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        # 使用 loadUi 函数加载 widget.ui 文件中的用户界面
        widget = loadUi('./widget.ui')
        self.widget = widget

        
        widget.bt_encipher.clicked.connect(self.encipher)
        widget.bt_decipher.clicked.connect(self.decipher)
        
        self.setCentralWidget(widget)
        
        self.setWindowTitle("S-AES")

    def encipher(self):
        
        plaintext =int( self.widget.plainTextEdit.text(),16)
        key=int(self.widget.keyEdit.text(),16)
        ciphertext = AES.sAES(plaintext, key)
        
        msgBox = QMessageBox()
        msgBox.setWindowTitle("加密成功")
        msgBox.setText('密文：'+hex(ciphertext))
        msgBox.exec_()  # 显示消息框

        

    def decipher(self):
        
        ciphertext = int(self.widget.cipherTextEdit.text(),16)
        key=int(self.widget.keyEdit.text(),16)
        plaintext=AES.invSAES(ciphertext,key)
        
        msgBox = QMessageBox()
        msgBox.setWindowTitle("解密成功")
        msgBox.setText('明文：'+hex(plaintext))
        msgBox.resize(200,200)
        msgBox.exec_()  # 显示消息框

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.resize(700,500)
    window.show()
    
    sys.exit(app.exec_())