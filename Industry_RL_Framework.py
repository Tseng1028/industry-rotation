
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import warnings
warnings.filterwarnings('ignore')

class IndustryTradingEnv(gym.Env):
    """
    Custom Environment for industry rotation trading
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, 
         features_dict: Dict[str, pd.DataFrame],
         cluster_data: pd.DataFrame,
         returns_data: pd.DataFrame,
         start_date: str,
         end_date: str,
         window_size: int = 60):

        super().__init__()
        
        # Data validation
        self._validate_data(features_dict, cluster_data, returns_data)
        
        # Store data
        self.features_dict = features_dict
        # 確保cluster_data的值為整數
        self.cluster_data = cluster_data.astype(int)
        self.returns_data = returns_data
        self.window_size = window_size
        
        # Convert dates
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Trading variables
        self.current_step = 0
        
        # 確保使用整數型別的群集標籤
        unique_clusters = np.unique(self.cluster_data.values.astype(int))
        self.n_clusters = len(unique_clusters)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_clusters)
        
        # Calculate total features
        total_features = sum(df.shape[1] for df in features_dict.values())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features * window_size,),
            dtype=np.float32
        )
        
        # Initialize dates
        self._initialize_dates()

    def _validate_data(self, features_dict, cluster_data, returns_data):
        if not isinstance(features_dict, dict):
            raise TypeError("features_dict must be a dictionary")
        
        for df in features_dict.values():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("All feature dataframes must have DatetimeIndex")
            # 增加多層column的處理
            if isinstance(df.columns, pd.MultiIndex):
                # 將多層column展平
                df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]

    def _initialize_dates(self):
        self.dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='B'
        )
        
    def _get_state(self) -> np.ndarray:
        current_date = self.dates[self.current_step]
        state_data = []
        
        for feature_df in self.features_dict.values():
            mask = (feature_df.index <= current_date) & \
                (feature_df.index > current_date - pd.Timedelta(days=self.window_size))
            # 如果是多層column的DataFrame，已在validate時展平
            historical = feature_df[mask].values
            
            if len(historical) < self.window_size:
                pad_length = self.window_size - len(historical)
                historical = np.pad(historical, ((pad_length, 0), (0, 0)), 
                                mode='constant', constant_values=0)
            
            state_data.append(historical.flatten())
        
        state = np.concatenate(state_data)
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_date = self.dates[self.current_step]
        
        next_dates = self.returns_data.index[
            self.returns_data.index > current_date]
        if len(next_dates) == 0:
            return self._get_state(), 0, True, False, {}
        next_date = next_dates[0]
        
        try:
            # 確保比較時使用相同的資料類型
            cluster_stocks = (self.cluster_data.loc[current_date].astype(int) == int(action))
            cluster_returns = self.returns_data.loc[next_date, cluster_stocks].mean()
            reward = float(cluster_returns) if not np.isnan(cluster_returns) else 0
            reward = np.clip(reward, -1, 1)
        except Exception as e:
            print(f"Error in step: {e}")
            reward = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        return self._get_state(), reward, done, False, {}
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_state(), {}

class SaveModelCallback(BaseCallback):
    """
    Callback for saving models during training
    """
    def __init__(self, save_path, save_freq=1000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'model_{self.n_calls}_steps')
            self.model.save(model_path)
        return True

def create_env(features_dict, cluster_data, returns_data, 
              start_date, end_date):
    """Create vectorized environment"""
    def make_env():
        env = IndustryTradingEnv(
            features_dict=features_dict,
            cluster_data=cluster_data,
            returns_data=returns_data,
            start_date=start_date,
            end_date=end_date
        )
        return env
    
    return DummyVecEnv([make_env])

def train_models(features_dict: Dict[str, pd.DataFrame], 
                cluster_data: pd.DataFrame,
                returns_data: pd.DataFrame,
                start_date: str,
                end_date: str,
                window_size: str = '6M',
                total_timesteps: int = 10000,
                models_dir: str = "./trained_models",
                force_retrain: bool = False,
                device: str = 'auto') -> pd.DataFrame:  
    try:
        # Input validation
        if not isinstance(features_dict, dict):
            raise ValueError("features_dict must be a dictionary")
        if not isinstance(cluster_data, pd.DataFrame):
            raise ValueError("cluster_data must be a pandas DataFrame")
        if not isinstance(returns_data, pd.DataFrame):
            raise ValueError("returns_data must be a pandas DataFrame")
            
        # Convert dates to pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 1. Data preprocessing
        print("Starting data preprocessing...")
        
        # 修改：只轉換值部分為整數,保持索引不變
        cluster_data_values = cluster_data.select_dtypes(include=['int', 'float']).astype(int)
        cluster_data = pd.DataFrame(
            cluster_data_values.values,
            index=cluster_data.index,
            columns=cluster_data_values.columns  
        )
        
        # Fill NaN values
        returns_data = returns_data.fillna(0)
        cluster_data = cluster_data.fillna(method='ffill')
        
        for key in features_dict:
            # Handle multi-level columns if present
            if isinstance(features_dict[key].columns, pd.MultiIndex):
                features_dict[key].columns = [f"{col[0]}_{col[1]}" for col in features_dict[key].columns]
            
            # Fill NaN values
            features_dict[key] = features_dict[key].fillna(method='ffill')
            
            # Standardize features
            mean = features_dict[key].mean()
            std = features_dict[key].std()
            features_dict[key] = (features_dict[key] - mean) / (std + 1e-8)

        # 3. Training loop
        results = []
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')

        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # 檢測 GPU 是否可用
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        print("Starting training loop...")
        for i in range(6, len(dates)-1):
            try:
                window_start = dates[i-6]
                window_end = dates[i]
                predict_date = dates[i+1]
                
                period_str = f"{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"
                print(f"\nTraining window: {window_start} to {window_end}, Predicting for: {predict_date}")
                
                # Create environment
                env = create_env(
                    features_dict=features_dict,
                    cluster_data=cluster_data,
                    returns_data=returns_data,
                    start_date=window_start.strftime('%Y-%m-%d'),
                    end_date=window_end.strftime('%Y-%m-%d')
                )
                
                period_results = {
                    'train_window_start': window_start,
                    'train_window_end': window_end,
                    'predict_date': predict_date
                }

                # 定義模型
                models = {
                    'A2C': lambda env: A2C(
                        'MlpPolicy',
                        env,
                        verbose=0,
                        learning_rate=1e-4,
                        gamma=0.99,
                        n_steps=5,
                        device=device
                    ),
                    'PPO': lambda env: PPO(
                        'MlpPolicy',
                        env,
                        verbose=0,
                        learning_rate=1e-4,
                        n_steps=2048,
                        batch_size=64,
                        device=device
                    ),
                    'TRPO': lambda env: TRPO(
                        'MlpPolicy',
                        env,
                        verbose=0,
                        gamma=0.99,
                        batch_size=64,
                        device=device
                    )
                }
                
                for model_name, model_builder in models.items():
                    try:
                        model_path = os.path.join(models_dir, model_name, f"model_{period_str}")
                        
                        if not force_retrain and os.path.exists(f"{model_path}.zip"):
                            print(f"Loading existing {model_name} model")
                            model = model_builder(env)
                            model = model.load(model_path, device=device)
                        else:
                            print(f"Training new {model_name} model")
                            model = model_builder(env)
                            
                            save_callback = SaveModelCallback(
                                save_path=os.path.join(models_dir, model_name),
                                save_freq=total_timesteps // 10
                            )
                            
                            model.learn(
                                total_timesteps=total_timesteps,
                                callback=save_callback,
                                progress_bar=True
                            )
                            
                            model.save(model_path)
                        
                        obs = env.reset()[0]
                        action, _ = model.predict(obs, deterministic=True, device=device)
                        period_results[f'{model_name}_prediction'] = int(action)
                        
                    except Exception as model_error:
                        print(f"Error with {model_name}: {str(model_error)}")
                        period_results[f'{model_name}_prediction'] = None
                
                results.append(period_results)
                
            except Exception as period_error:
                print(f"Error processing period: {str(period_error)}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Fatal error in train_models: {str(e)}")
        raise

def load_trained_model(model_type: str, 
                      period_start: str, 
                      period_end: str, 
                      models_dir: str = "./trained_models"):
    """
    加載已訓練的模型
    
    Parameters:
    - model_type: 模型類型 ('A2C', 'PPO', 或 'TRPO')
    - period_start: 訓練期間開始日期 (YYYYMMDD)
    - period_end: 訓練期間結束日期 (YYYYMMDD)
    - models_dir: 模型保存目錄
    """
    models = {
        'A2C': A2C,
        'PPO': PPO,
        'TRPO': TRPO
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_path = os.path.join(
        models_dir, 
        model_type, 
        f"model_{period_start}_{period_end}"
    )
    
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"No trained model found at {model_path}")
    
    return models[model_type].load(model_path)

def initialize_gpu_settings():
    """Safely initialize GPU settings"""
    try:
        if torch.cuda.is_available():
            # Set device
            device = torch.device('cuda')
            
            # Configure CUDA settings for stability
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Set default tensor type
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            
            print(f"Using GPU: {gpu_name}")
            print(f"Initial GPU Memory: Allocated: {memory_allocated/1e6:.2f}MB, Reserved: {memory_reserved/1e6:.2f}MB")
            
            return 'cuda'
        else:
            print("No GPU available, using CPU")
            return 'cpu'
    except Exception as e:
        print(f"Error initializing GPU: {str(e)}")
        print("Falling back to CPU")
        return 'cpu'

def safe_train_models(features_dict, cluster_data, returns_data,
                     start_date, end_date, window_size='6M',
                     total_timesteps=10000, models_dir="./trained_models",
                     force_retrain=False):
    """
    Safer version of train_models with robust GPU handling
    """
    try:
        # Initialize device settings
        device = initialize_gpu_settings()
        print(f"Using device: {device}")
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Ensure data is in float32 for numerical stability
        features_dict = {k: v.astype('float32') for k, v in features_dict.items()}
        returns_data = returns_data.astype('float32')
        
        # Initialize results storage
        results = []
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        for i in range(6, len(dates)-1):
            try:
                window_start = dates[i-6]
                window_end = dates[i]
                predict_date = dates[i+1]
                
                print(f"\nProcessing period: {window_start} to {window_end}")
                
                # Create environment
                env_kwargs = {
                    'features_dict': {k: v.loc[window_start:window_end] for k, v in features_dict.items()},
                    'cluster_data': cluster_data.loc[window_start:window_end],
                    'returns_data': returns_data.loc[window_start:window_end],
                    'start_date': window_start.strftime('%Y-%m-%d'),
                    'end_date': window_end.strftime('%Y-%m-%d')
                }
                
                # Create vectorized environment safely
                try:
                    env = create_env(**env_kwargs)
                    print("Environment created successfully")
                except Exception as env_error:
                    print(f"Error creating environment: {str(env_error)}")
                    continue
                
                period_results = {
                    'train_window_start': window_start,
                    'train_window_end': window_end,
                    'predict_date': predict_date
                }
                
                # Define model configurations with safe defaults
                model_configs = {
                    'A2C': {
                        'policy': 'MlpPolicy',
                        'learning_rate': 1e-4,
                        'n_steps': 5,
                        'gamma': 0.99,
                        'verbose': 0,
                        'device': device,
                        'tensorboard_log': None
                    },
                    'PPO': {
                        'policy': 'MlpPolicy',
                        'learning_rate': 1e-4,
                        'n_steps': 2048,
                        'batch_size': 64,
                        'verbose': 0,
                        'device': device,
                        'tensorboard_log': None
                    },
                    'TRPO': {
                        'policy': 'MlpPolicy',
                        'learning_rate': 1e-4,
                        'batch_size': 64,
                        'verbose': 0,
                        'device': device,
                        'tensorboard_log': None
                    }
                }
                
                for model_name, config in model_configs.items():
                    try:
                        print(f"\nTraining {model_name}...")
                        
                        # Safely create and train model
                        if model_name == 'A2C':
                            model = A2C(env=env, **config)
                        elif model_name == 'PPO':
                            model = PPO(env=env, **config)
                        else:  # TRPO
                            model = TRPO(env=env, **config)
                        
                        # Train with error handling
                        try:
                            model.learn(
                                total_timesteps=total_timesteps,
                                progress_bar=True
                            )
                            print(f"{model_name} training completed")
                        except Exception as train_error:
                            print(f"Error during {model_name} training: {str(train_error)}")
                            continue
                        
                        # Make prediction
                        try:
                            obs = env.reset()[0]
                            action, _ = model.predict(obs, deterministic=True)
                            period_results[f'{model_name}_prediction'] = int(action)
                            print(f"{model_name} prediction: {action}")
                        except Exception as pred_error:
                            print(f"Error making prediction with {model_name}: {str(pred_error)}")
                            period_results[f'{model_name}_prediction'] = None
                        
                        # Clean up
                        del model
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as model_error:
                        print(f"Error with {model_name}: {str(model_error)}")
                        period_results[f'{model_name}_prediction'] = None
                
                results.append(period_results)
                
            except Exception as period_error:
                print(f"Error processing period: {str(period_error)}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Fatal error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 使用示例
if __name__ == "__main__":
    """
    # 準備數據
    features_dict = {...}  # 特徵數據字典
    cluster_data = ...     # 群集數據
    returns_data = ...     # 報酬率數據
    
    # 執行訓練和預測
    results = train_models(
        features_dict=features_dict,
        cluster_data=cluster_data,
        returns_data=returns_data,
        start_date='2020-01-01',
        end_date='2023-12-31',
        window_size='6M',
        models_dir='./my_models',
        force_retrain=False
    )
    
    print(results)
    """
    pass