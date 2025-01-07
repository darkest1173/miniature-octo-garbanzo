1.准备输入图像
将一张想要进行插值变换的图像(命名为 input.jpg)放到与 nearest_neighbor.cpp 相同的工作目录下，或者在代码中修改为图像的绝对路径。
2.编译代码
使用 g++ 命令行，并已安装了 OpenCV :
g++ nearest_neighbor.cpp -o nearest_neighbor
pkg-config --cflags --libs opencv4
这条命令会：
• 将 nearest_neighbor.cpp 编译为可执行文件 nearest_neighbor。
• pkg-config --cflags --libs opencv4 会自动填充编译和链接参数(针对 OpenCV 4.x)。
3.运行可执行文件
在终端执行:
./nearest_neighbor
4.程序将会：
• 读入 input.jpg
• 应用最近邻插值将其缩放到 256×256(可以在代码中自行修改输出尺寸)。
• 在窗口中显示原图和插值后的结果，并在同文件夹下生成一张名为 output.jpg 的输出图像。
5.查看结果
• 等窗口显示处理完成的图像后，按任意键退出。
• 在程序目录下找到 output.jpg，即为最近邻插值处理后的图像
