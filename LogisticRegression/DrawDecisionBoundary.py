import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model,axis):
    """
    绘制机器学习模型的决策边界
    
    该函数通过在指定区域内生成密集的网格点，对每个点进行预测，
    然后根据预测结果绘制不同类别的区域，从而可视化模型的决策边界
    
    参数:
    model: 已训练的机器学习模型，必须具有predict()方法
    axis: 绘图区域的边界，格式为[x_min, x_max, y_min, y_max]
    """
    
    # 创建网格坐标矩阵
    # np.meshgrid()生成坐标网格，用于创建决策边界的采样点
    x0,x1=np.meshgrid(
        # 在x轴方向生成均匀分布的点
        # int((axis[1]-axis[0])*100) 根据坐标范围确定点的密度
        # 密度越高，决策边界越平滑，但计算量也越大
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        # 在y轴方向生成均匀分布的点
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    
    # 将网格坐标转换为模型输入格式
    # x0.ravel() 将二维网格展平为一维数组（x坐标）
    # x1.ravel() 将二维网格展平为一维数组（y坐标）
    # np.c_[] 按列拼接，创建形状为(n_points, 2)的特征矩阵
    # 每行代表一个网格点的坐标[x, y]
    X_new=np.c_[x0.ravel(),x1.ravel()]
    
    # 使用训练好的模型对所有网格点进行预测
    # 获得每个点的类别预测结果
    y_predict=model.predict(X_new)
    
    # 将一维预测结果重新整形为二维网格形状
    # 这样可以与原始的x0, x1网格对应，便于绘制等高线图
    zz=y_predict.reshape(x0.shape)
    
    # 导入自定义颜色映射
    from matplotlib.colors import ListedColormap
    # 创建自定义颜色映射：
    # '#EF9A9A' - 浅红色，通常用于类别0
    # '#FFF59D' - 浅黄色，通常用于类别1  
    # '#90CAF9' - 浅蓝色，通常用于类别2（如果有第三类）
    custom_cmap=ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    
    # 绘制填充等高线图（决策区域）
    # plt.contourf() 创建填充的等高线图，不同颜色区域代表不同的预测类别
    # x0, x1: 网格坐标
    # zz: 对应的预测值
    # cmap: 颜色映射，决定不同类别使用的颜色
    plt.contourf(x0,x1,zz,cmap=custom_cmap)
