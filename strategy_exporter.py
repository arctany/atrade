import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import os
import yaml
import pickle
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

class StrategyExporter:
    def __init__(self, strategy_data: Dict):
        """
        初始化策略导出器
        
        Args:
            strategy_data: 策略数据
        """
        self.strategy_data = strategy_data
        
    def export_strategy(
        self,
        strategy_name: str,
        export_format: str = 'json',
        output_dir: str = 'exported_strategies'
    ) -> Dict:
        """
        导出策略
        
        Args:
            strategy_name: 策略名称
            export_format: 导出格式 (json, yaml, pickle)
            output_dir: 输出目录
            
        Returns:
            导出结果
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备导出数据
            export_data = self._prepare_export_data(strategy_name)
            
            # 根据格式导出
            if export_format == 'json':
                output_path = self._export_json(export_data, strategy_name, output_dir)
            elif export_format == 'yaml':
                output_path = self._export_yaml(export_data, strategy_name, output_dir)
            elif export_format == 'pickle':
                output_path = self._export_pickle(export_data, strategy_name, output_dir)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # 创建策略包
            package_path = self._create_strategy_package(
                strategy_name,
                output_path,
                output_dir
            )
            
            return {
                'status': 'success',
                'strategy_name': strategy_name,
                'export_format': export_format,
                'output_path': output_path,
                'package_path': package_path,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting strategy: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _prepare_export_data(self, strategy_name: str) -> Dict:
        """准备导出数据"""
        try:
            strategy = self.strategy_data[strategy_name]
            
            export_data = {
                'strategy_name': strategy_name,
                'parameters': strategy['parameters'],
                'performance_metrics': strategy['performance_metrics'],
                'trades': strategy['trades'],
                'equity_curve': strategy['equity_curve'],
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'description': f"Exported strategy: {strategy_name}"
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error preparing export data: {str(e)}")
            raise
    
    def _export_json(self, data: Dict, strategy_name: str, output_dir: str) -> str:
        """导出为JSON格式"""
        try:
            output_path = os.path.join(output_dir, f"{strategy_name}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise
    
    def _export_yaml(self, data: Dict, strategy_name: str, output_dir: str) -> str:
        """导出为YAML格式"""
        try:
            output_path = os.path.join(output_dir, f"{strategy_name}.yaml")
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to YAML: {str(e)}")
            raise
    
    def _export_pickle(self, data: Dict, strategy_name: str, output_dir: str) -> str:
        """导出为Pickle格式"""
        try:
            output_path = os.path.join(output_dir, f"{strategy_name}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to Pickle: {str(e)}")
            raise
    
    def _create_strategy_package(
        self,
        strategy_name: str,
        strategy_file: str,
        output_dir: str
    ) -> str:
        """创建策略包"""
        try:
            # 创建策略包目录
            package_dir = os.path.join(output_dir, f"{strategy_name}_package")
            os.makedirs(package_dir, exist_ok=True)
            
            # 复制策略文件
            strategy_filename = os.path.basename(strategy_file)
            package_file = os.path.join(package_dir, strategy_filename)
            with open(strategy_file, 'rb') as src, open(package_file, 'wb') as dst:
                dst.write(src.read())
            
            # 创建README文件
            readme_content = f"""
            # {strategy_name} Strategy Package
            
            This package contains the exported trading strategy.
            
            ## Contents
            - {strategy_filename}: Strategy data file
            - README.md: This file
            
            ## Usage
            1. Import the strategy file using the appropriate method for your format
            2. Load the strategy data
            3. Use the strategy in your trading system
            
            ## Metadata
            - Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Version: 1.0.0
            """
            
            with open(os.path.join(package_dir, 'README.md'), 'w') as f:
                f.write(readme_content)
            
            # 创建ZIP文件
            zip_path = os.path.join(output_dir, f"{strategy_name}_package.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir)
                        zipf.write(file_path, arcname)
            
            return zip_path
            
        except Exception as e:
            logger.error(f"Error creating strategy package: {str(e)}")
            raise
    
    def export_multiple_strategies(
        self,
        strategy_names: List[str],
        export_format: str = 'json',
        output_dir: str = 'exported_strategies'
    ) -> Dict:
        """
        导出多个策略
        
        Args:
            strategy_names: 策略名称列表
            export_format: 导出格式
            output_dir: 输出目录
            
        Returns:
            导出结果
        """
        try:
            results = []
            for strategy_name in strategy_names:
                result = self.export_strategy(
                    strategy_name,
                    export_format,
                    output_dir
                )
                results.append(result)
            
            return {
                'status': 'success',
                'results': results,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting multiple strategies: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def export_strategy_template(
        self,
        template_name: str,
        output_dir: str = 'strategy_templates'
    ) -> Dict:
        """
        导出策略模板
        
        Args:
            template_name: 模板名称
            output_dir: 输出目录
            
        Returns:
            导出结果
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建模板数据
            template_data = {
                'strategy_name': template_name,
                'parameters': {
                    'param1': {
                        'type': 'float',
                        'default': 0.0,
                        'description': 'Parameter 1'
                    },
                    'param2': {
                        'type': 'int',
                        'default': 0,
                        'description': 'Parameter 2'
                    }
                },
                'indicators': {
                    'indicator1': {
                        'type': 'moving_average',
                        'parameters': {
                            'window': 20
                        }
                    }
                },
                'signals': {
                    'buy': {
                        'conditions': [],
                        'description': 'Buy signal conditions'
                    },
                    'sell': {
                        'conditions': [],
                        'description': 'Sell signal conditions'
                    }
                },
                'risk_management': {
                    'position_size': {
                        'type': 'fixed',
                        'value': 1.0
                    },
                    'stop_loss': {
                        'type': 'percentage',
                        'value': 0.02
                    }
                },
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'description': f"Template for {template_name} strategy"
                }
            }
            
            # 导出模板
            output_path = os.path.join(output_dir, f"{template_name}_template.json")
            with open(output_path, 'w') as f:
                json.dump(template_data, f, indent=4)
            
            return {
                'status': 'success',
                'template_name': template_name,
                'output_path': output_path,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting strategy template: {str(e)}")
            return {'status': 'error', 'message': str(e)} 