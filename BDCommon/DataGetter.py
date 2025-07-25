import numpy as np
from sklearn.datasets import fetch_openml

class BostonData:

    def __init__(self):
        self.data=None
        self.target=None

    def get_boston(self,need_clean=True):
        # 加载数据
        boston = fetch_openml(name='boston', version=1, as_frame=False)
        
        # 清理特征数据
        self.data= boston.data
        self.target = boston.target

        if need_clean:
            self.data=self.clean_boston_data()
            self.data,self.target=self.clean_boston_target()

    def clean_boston_data(self):
        """
        专门处理Boston数据集的数据清理
        """
        boston_data=self.data
        # 确保输入是numpy数组
        if not isinstance(boston_data, np.ndarray):
            boston_data = np.array(boston_data)
        
        print(f"原始数据形状: {boston_data.shape}")
        print(f"原始数据类型: {boston_data.dtype}")
        
        # 如果是object类型，需要逐列检查
        if boston_data.dtype == 'object':
            numeric_cols = []
            for i in range(boston_data.shape[1]):
                col = boston_data[:, i]
                try:
                    # 尝试转换为float
                    numeric_col = np.array(col, dtype=np.float64)
                    # 检查是否有有效的数值
                    if not np.all(np.isnan(numeric_col)):
                        numeric_cols.append(numeric_col)
                    else:
                        print(f"列 {i} 全部为NaN，已剔除")
                except (ValueError, TypeError) as e:
                    print(f"列 {i} 无法转换为数值类型: {e}")
                    continue
            
            if numeric_cols:
                # 重新组合数值列
                cleaned_data = np.column_stack(numeric_cols)
                print(f"清理后数据形状: {cleaned_data.shape}")
                return cleaned_data
            else:
                raise ValueError("没有找到有效的数值列")
        else:
            # 如果已经是数值类型，直接返回
            return boston_data

    def clean_boston_target(self):
        
        try:
            y = np.array(self.target, dtype=np.float64)
            # 移除NaN值
            valid_y_mask = ~np.isnan(y)
            y = y[valid_y_mask]
            x = self.data[valid_y_mask]  # 保持x和y的对应关系
            return x,y
            
        except Exception as e:
            print(f"目标数据清理失败: {e}")
            raise