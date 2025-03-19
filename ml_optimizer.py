import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import os
import json
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import ML_CONFIG

# 设置日志
logger = logging.getLogger(__name__)

class MLOptimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        features: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        初始化机器学习优化器
        
        Args:
            data: 输入数据
            target: 目标变量
            features: 特征列表
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data = data
        self.target = target
        self.features = features
        self.test_size = test_size
        self.random_state = random_state
        self.ml_config = ML_CONFIG
        
        # 初始化结果存储
        self.models = {}
        self.optimization_history = {}
        self.best_models = {}
        
        # 确保GPU可用性
        self._setup_gpu()
        
        # 准备数据
        self._prepare_data()
    
    def _setup_gpu(self):
        """设置GPU环境"""
        try:
            # 检查是否有可用的GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # 设置GPU内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s)")
            else:
                logger.warning("No GPU found, using CPU")
        except Exception as e:
            logger.warning(f"Error setting up GPU: {str(e)}")
    
    def _prepare_data(self):
        """准备数据"""
        try:
            # 分离特征和目标
            X = self.data[self.features]
            y = self.data[self.target]
            
            # 标准化特征
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 划分训练集和测试集
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            logger.info("Data preparation completed")
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def optimize_neural_network(
        self,
        n_trials: int = 100,
        max_epochs: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """
        优化神经网络模型
        
        Args:
            n_trials: 优化试验次数
            max_epochs: 最大训练轮数
            batch_size: 批次大小
            
        Returns:
            优化结果
        """
        try:
            def objective(trial):
                # 定义超参数搜索空间
                params = {
                    'n_layers': trial.suggest_int('n_layers', 1, 4),
                    'n_units': trial.suggest_int('n_units', 32, 256),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
                }
                
                # 构建模型
                model = self._build_neural_network(params)
                
                # 训练模型
                history = model.fit(
                    self.X_train, self.y_train,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # 计算验证集损失
                val_loss = min(history.history['val_loss'])
                
                return val_loss
            
            # 创建优化研究
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # 使用最佳参数训练最终模型
            best_params = study.best_params
            best_model = self._build_neural_network(best_params)
            best_model.fit(
                self.X_train, self.y_train,
                epochs=max_epochs,
                batch_size=batch_size,
                validation_split=0.2
            )
            
            # 评估模型
            metrics = self._evaluate_model(best_model)
            
            # 保存结果
            self.models['neural_network'] = best_model
            self.optimization_history['neural_network'] = study.trials_dataframe()
            self.best_models['neural_network'] = {
                'model': best_model,
                'params': best_params,
                'metrics': metrics
            }
            
            return {
                'best_params': best_params,
                'metrics': metrics,
                'optimization_history': study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing neural network: {str(e)}")
            raise
    
    def optimize_xgboost(
        self,
        n_trials: int = 100
    ) -> Dict:
        """
        优化XGBoost模型
        
        Args:
            n_trials: 优化试验次数
            
        Returns:
            优化结果
        """
        try:
            def objective(trial):
                # 定义超参数搜索空间
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9)
                }
                
                # 构建模型
                model = xgb.XGBRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                # 训练模型
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_test, self.y_test)],
                    early_stopping_rounds=50,
                    verbose=0
                )
                
                # 计算验证集损失
                val_loss = model.evals_result()['validation_0']['rmse'][-1]
                
                return val_loss
            
            # 创建优化研究
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # 使用最佳参数训练最终模型
            best_params = study.best_params
            best_model = xgb.XGBRegressor(
                **best_params,
                random_state=self.random_state,
                n_jobs=-1
            )
            best_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=50
            )
            
            # 评估模型
            metrics = self._evaluate_model(best_model)
            
            # 保存结果
            self.models['xgboost'] = best_model
            self.optimization_history['xgboost'] = study.trials_dataframe()
            self.best_models['xgboost'] = {
                'model': best_model,
                'params': best_params,
                'metrics': metrics
            }
            
            return {
                'best_params': best_params,
                'metrics': metrics,
                'optimization_history': study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing XGBoost: {str(e)}")
            raise
    
    def optimize_lightgbm(
        self,
        n_trials: int = 100
    ) -> Dict:
        """
        优化LightGBM模型
        
        Args:
            n_trials: 优化试验次数
            
        Returns:
            优化结果
        """
        try:
            def objective(trial):
                # 定义超参数搜索空间
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9)
                }
                
                # 构建模型
                model = lgb.LGBMRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                # 训练模型
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_test, self.y_test)],
                    early_stopping_rounds=50,
                    verbose=0
                )
                
                # 计算验证集损失
                val_loss = model.evals_result()['valid_0']['rmse'][-1]
                
                return val_loss
            
            # 创建优化研究
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # 使用最佳参数训练最终模型
            best_params = study.best_params
            best_model = lgb.LGBMRegressor(
                **best_params,
                random_state=self.random_state,
                n_jobs=-1
            )
            best_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=50
            )
            
            # 评估模型
            metrics = self._evaluate_model(best_model)
            
            # 保存结果
            self.models['lightgbm'] = best_model
            self.optimization_history['lightgbm'] = study.trials_dataframe()
            self.best_models['lightgbm'] = {
                'model': best_model,
                'params': best_params,
                'metrics': metrics
            }
            
            return {
                'best_params': best_params,
                'metrics': metrics,
                'optimization_history': study.trials_dataframe()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing LightGBM: {str(e)}")
            raise
    
    def _build_neural_network(self, params: Dict) -> tf.keras.Model:
        """构建神经网络模型"""
        try:
            model = tf.keras.Sequential()
            
            # 添加输入层
            model.add(tf.keras.layers.Dense(
                units=params['n_units'],
                activation='relu',
                input_shape=(self.X_train.shape[1],)
            ))
            
            # 添加隐藏层
            for _ in range(params['n_layers']):
                model.add(tf.keras.layers.Dense(
                    units=params['n_units'],
                    activation='relu'
                ))
                model.add(tf.keras.layers.Dropout(
                    rate=params['dropout_rate']
                ))
            
            # 添加输出层
            model.add(tf.keras.layers.Dense(units=1))
            
            # 编译模型
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=params['learning_rate']
                ),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building neural network: {str(e)}")
            raise
    
    def _evaluate_model(self, model: Union[tf.keras.Model, xgb.XGBRegressor, lgb.LGBMRegressor]) -> Dict:
        """评估模型性能"""
        try:
            # 预测
            if isinstance(model, tf.keras.Model):
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_test)
            
            # 计算评估指标
            mse = np.mean((self.y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(self.y_test - y_pred))
            r2 = 1 - np.sum((self.y_test - y_pred) ** 2) / np.sum((self.y_test - np.mean(self.y_test)) ** 2)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_models(self, output_dir: str = 'saved_models'):
        """保存模型"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存每个模型
            for name, model_info in self.best_models.items():
                model_dir = os.path.join(output_dir, name)
                os.makedirs(model_dir, exist_ok=True)
                
                # 保存模型
                if isinstance(model_info['model'], tf.keras.Model):
                    model_info['model'].save(os.path.join(model_dir, 'model.h5'))
                elif isinstance(model_info['model'], xgb.XGBRegressor):
                    model_info['model'].save_model(os.path.join(model_dir, 'model.json'))
                elif isinstance(model_info['model'], lgb.LGBMRegressor):
                    model_info['model'].booster_.save_model(os.path.join(model_dir, 'model.txt'))
                
                # 保存参数和指标
                with open(os.path.join(model_dir, 'info.json'), 'w') as f:
                    json.dump({
                        'params': model_info['params'],
                        'metrics': model_info['metrics']
                    }, f, indent=4)
            
            logger.info(f"Models saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, input_dir: str = 'saved_models'):
        """加载模型"""
        try:
            # 检查输入目录
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Directory not found: {input_dir}")
            
            # 加载每个模型
            for name in os.listdir(input_dir):
                model_dir = os.path.join(input_dir, name)
                if not os.path.isdir(model_dir):
                    continue
                
                # 加载模型
                model_path = os.path.join(model_dir, 'model.h5' if name == 'neural_network' else 'model.json' if name == 'xgboost' else 'model.txt')
                if not os.path.exists(model_path):
                    continue
                
                if name == 'neural_network':
                    model = tf.keras.models.load_model(model_path)
                elif name == 'xgboost':
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                elif name == 'lightgbm':
                    model = lgb.LGBMRegressor()
                    model.booster_ = lgb.Booster(model_file=model_path)
                
                # 加载参数和指标
                with open(os.path.join(model_dir, 'info.json'), 'r') as f:
                    info = json.load(f)
                
                # 保存到结果存储
                self.models[name] = model
                self.best_models[name] = {
                    'model': model,
                    'params': info['params'],
                    'metrics': info['metrics']
                }
            
            logger.info(f"Models loaded from {input_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def generate_optimization_report(
        self,
        output_dir: str = 'optimization_reports',
        include_charts: bool = True,
        include_tables: bool = True
    ) -> str:
        """
        生成优化报告
        
        Args:
            output_dir: 输出目录
            include_charts: 是否包含图表
            include_tables: 是否包含表格
            
        Returns:
            报告文件路径
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成HTML报告
            html_content = self._generate_html_report(include_charts, include_tables)
            
            # 保存报告
            report_path = os.path.join(output_dir, f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {str(e)}")
            raise
    
    def _generate_html_report(self, include_charts: bool, include_tables: bool) -> str:
        """生成HTML报告"""
        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>ML Model Optimization Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin-bottom: 30px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .chart { margin-bottom: 30px; }
                </style>
            </head>
            <body>
                <h1>ML Model Optimization Report</h1>
                <p>Generated at: {timestamp}</p>
            """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            if include_charts:
                html_content += self._generate_charts()
            
            if include_tables:
                html_content += self._generate_tables()
            
            html_content += """
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def _generate_charts(self) -> str:
        """生成图表"""
        try:
            # 创建优化历史图表
            optimization_fig = go.Figure()
            for name, history in self.optimization_history.items():
                optimization_fig.add_trace(go.Scatter(
                    x=history.index,
                    y=history['value'],
                    name=name,
                    mode='lines+markers'
                ))
            optimization_fig.update_layout(
                title='Optimization History',
                xaxis_title='Trial',
                yaxis_title='Loss'
            )
            optimization_html = optimization_fig.to_html(full_html=False)
            
            # 创建模型比较图表
            comparison_fig = go.Figure()
            metrics = ['mse', 'rmse', 'mae', 'r2']
            for metric in metrics:
                values = [model_info['metrics'][metric] for model_info in self.best_models.values()]
                comparison_fig.add_trace(go.Bar(
                    x=list(self.best_models.keys()),
                    y=values,
                    name=metric
                ))
            comparison_fig.update_layout(
                title='Model Comparison',
                xaxis_title='Model',
                yaxis_title='Value'
            )
            comparison_html = comparison_fig.to_html(full_html=False)
            
            return f"""
                <div class="section">
                    <h2>Charts</h2>
                    <div class="chart">
                        {optimization_html}
                    </div>
                    <div class="chart">
                        {comparison_html}
                    </div>
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            raise
    
    def _generate_tables(self) -> str:
        """生成表格"""
        try:
            # 生成模型参数表格
            params_table = pd.DataFrame({
                name: model_info['params']
                for name, model_info in self.best_models.items()
            }).to_html()
            
            # 生成模型指标表格
            metrics_table = pd.DataFrame({
                name: model_info['metrics']
                for name, model_info in self.best_models.items()
            }).to_html()
            
            return f"""
                <div class="section">
                    <h2>Tables</h2>
                    <h3>Model Parameters</h3>
                    {params_table}
                    <h3>Model Metrics</h3>
                    {metrics_table}
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating tables: {str(e)}")
            raise 