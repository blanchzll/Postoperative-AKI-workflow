# AKI全部主图与附图独立绘图包

当前范围为2026年11月11日投稿版本的5幅主图和26幅附图。全部绘图汇总数据位于data，代码位于code。图号与代码、输入表的对应清单见FIGURE_CODE_INDEX.csv。reference保存当前投稿图和图注，供重绘比较。

## 命令行运行

使用Python3.12，安装requirements.txt后，在本目录运行以下命令。

    python run_figures.py

只生成指定图片可运行以下命令。

    python run_figures.py --only Figure_1 eFigure_25

默认输出位于outputs。主图在outputs/main，附图完整PDF在outputs/supplement。附图的原尺寸分页PDF、可编辑SVG、300dpi PNG和600dpi TIFF在outputs/supplement/pages。每幅附图最多四面板，部分附图分两页保存。

程序会自动查找Arial。macOS通常可直接找到系统补充字体。需要手工指定时使用以下参数。

    python run_figures.py --font-dir "/System/Library/Fonts/Supplemental"

本包采用原图Arial字体参数和原有JAMA配色。字体由运行系统提供。

## Notebook运行

安装requirements-notebook.txt并打开notebooks/01_Reproduce_All_Figures.ipynb。服务器已配置AKI Figures (Python 3.12)内核。按顺序运行单元格，可重绘全部图并在Notebook中预览。02_Select_One_Figure.ipynb可选择单幅图片，直接编辑代码后重绘。

输入为封存汇总结果，运行过程读取这些结果并绘图。模型开发、患者级预测和bootstrap计算的研究代码随完整投稿包保存。

verification记录独立运行测试、PDF逐页比较和数据来源哈希。运行结束后绘图子进程退出，Jupyter仅保留你正在使用的交互内核。

## 本次图形调整

Figure1A按作者要求替换为四组配对比较中八条选定流程的AUROC结果条形图，显示原始95%区间和人群数量。Figure1B-D延用对应配对比较和捕获结果。eFigure8B采用当前源码中的轴标签，去除了封存图上重复的连续结局纵轴标题。验证报告逐项记录这些呈现差异。
