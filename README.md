# Feature-Match-with-Surf
Feature Match with Surf, opencv-contrib-python.

requirements:
python2 or python3
opencv-python==3.4.3.18
opencv-contrib-python==3.4.1.15
numpy

excute:
python surf_match.py


反光合入代码环境配置：
1.查看～/.local/lib/python2.7/site-packages文件夹下是否存在opencv_contrib_python-3.4.1.15.dist-info、opencv_python-3.4.3.18.dist-info、cv2文件夹。
(代码依赖opencv两个库：opencv-python-3.4.3.18和opencv-contrib-python-3.4.1.15， 之前已在机器人上通过如下命令安装完毕，安装好了就可以不用再装：
pip install -i http://mirrors.aliyun.com/pypi/simple/ opencv-python==3.4.3.18     (如果http失败就用https)
pip install -i http://mirrors.aliyun.com/pypi/simple/ opencv-contrib-python==3.4.1.15)  (如果http失败就用https)
2.在～/.bashrc文件中最后加入：
export PYTHONPATH=/home/rick/.local/lib/python2.7/site-packages:$PYTHONPATH
确保该路径在环境变量PYTHONPATH里的ros部分之前。注意：supervisor启动也需要添加此环境变量。
3.命令行输入：. ~/.bashrc 使修改生效
4.命令行输入：
python
import cv2
cv2.TrackerCSRT_create()
看是否报错以作为验证
如果不成功先看看版本号
cv2.__version__
如果该语句执行也有问题
则卸载后重新安装：
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip list 查看是否将opencv卸载干净，如果还有就继续卸载
重装
5.将附件中的yhy.zip解压到home目录下
命令行输入：
cd ～/yhy
python surf_match.py
观察是否打印报错，如果无报错且打印一系列test done则表示成功配置
6.将～/yhy目录中的surf_match.py替换原文件（注意：备份原文件）
7.预置位标注时注意让方框刚好包罗图标，上下左右不要留出太多空白区域。

