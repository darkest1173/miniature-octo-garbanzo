#include <QApplication>
#include "ImageResizer.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    
    // 设置默认编码为UTF-8
    #if (QT_VERSION >= QT_VERSION_CHECK(5,0,0))
        QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
    #endif
    
    // 设置应用程序属性
    QApplication::setApplicationName("图像处理器");
    QApplication::setApplicationDisplayName("图像处理器");
    
    ImageResizer w;
    w.show();
    return a.exec();
}