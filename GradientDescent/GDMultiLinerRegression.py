import numpy as np
from sklearn.metrics import r2_score

def clean_boston_data(boston_data):
    """
    专门处理Boston数据集的数据清理
    """
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


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor
    
    try:
        # 加载数据
        boston = fetch_openml(name='boston', version=1, as_frame=False)
        
        # 清理特征数据
        x_raw = boston.data
        y_raw = boston.target
        
        print("开始清理特征数据...")
        x = clean_boston_data(x_raw)
        
        # 清理目标数据
        print("开始清理目标数据...")
        try:
            y = np.array(y_raw, dtype=np.float64)
            # 移除NaN值
            valid_y_mask = ~np.isnan(y)
            y = y[valid_y_mask]
            x = x[valid_y_mask]  # 保持x和y的对应关系
        except Exception as e:
            print(f"目标数据清理失败: {e}")
            raise
        
        print(f"最终数据形状 - X: {x.shape}, Y: {y.shape}")
        
        # 过滤数据
        mask = y < 25
        X = x[mask]
        Y = y[mask]
        
        print(f"过滤后数据形状 - X: {X.shape}, Y: {Y.shape}")
        
        # 继续训练模型
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        standard=StandardScaler()
        standard.fit(X_train)
        X_train_standard=standard.transform(X_train)
        X_test_standard=standard.transform(X_test)
        
        reg=SGDRegressor()
        reg.fit(X_train_standard,y_train)
        y_pred=reg.predict(X_test_standard)
        print("Coefficients:", reg.coef_)
        print("Interception:", reg.intercept_)
        r2=r2_score(y_test,y_pred)
        print(f"R2 Score: {r2:.4f}")

        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()