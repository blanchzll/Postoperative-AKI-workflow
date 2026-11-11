# WorkBuddy规格书对应的AKI绘图包V6

总入口为run_figures.py。主图1至5的20个面板由code/draw_main_workbuddy.py生成。完整图号、代码和输入对应表见FIGURE_CODE_INDEX.csv。94个冻结汇总输入保存在data。默认生成5幅主图和26幅附图。

    python run_figures.py
    python run_figures.py --only Figure_1 Figure_2 Figure_3 Figure_4 Figure_5

服务器Python环境沿用AKI Figures (Python3.12)，代码与Notebook均在本目录。Notebook为notebooks/01_Reproduce_All_Figures.ipynb。服务器图片输出为/home/lei/codex-workspaces/aki-prediction-manuscript/outputs/workbuddy_figures_20261111_v6，main存放主图，supplement存放附图，附图原尺寸分页在supplement/pages。新图由作者自行复制到电脑。

主图采用171mm页宽、至少8pt文字、Arial字体和蓝/朱红评价配色。1D捕获计数与配对差采用分开的坐标。3D使用规格书允许的点估计矩阵，区间保留在补表。4B的Brier下降量明确乘1000展示。5B横轴扩至0.115以容纳原始区间上限0.1072，5D采用0至80%，完整包含所展示的Wilson区间。

PDF和可编辑SVG为矢量输出，同时提供300dpi PNG和600dpi LZW TIFF。附图沿用原数据到图形映射。运行测试与数值、字体、边界核验记录在verification/WORKBUDDY_V6_ACCEPTANCE.json。中英文图注位于reference。
