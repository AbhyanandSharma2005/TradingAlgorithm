# TradingAlgorithm

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Deque
import statistics
import collections
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

class Trader:
    def __init__(self):
        self.price_history: Dict[str, Deque[float]] = collections.defaultdict(lambda: collections.deque(maxlen=20))  # Store last 20 prices
        self.trade_history: Dict[str, List[float]] = collections.defaultdict(list)  # Store past successful trades
        self.ema_values: Dict[str, float] = {}  # Exponential moving averages
        self.alpha = 0.2  # Smoothing factor for EMA
        self.max_drawdown_threshold = 5.0  # Maximum acceptable loss before reducing trade size
        self.model = LinearRegression()  # Price prediction model
        
        # Deep Q-Network (DQN) parameters
        self.memory = deque(maxlen=1000)
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration_rate = 1.0  # Start with high exploration
        self.exploration_decay = 0.995
        self.exploration_min = 0.01
        self.dqn_model = self.build_dqn()
    
    def build_dqn(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(3,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='linear')  # Outputs Q-values for buy, sell, hold
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def calculate_ema(self, product: str, new_price: float):
        if product not in self.ema_values:
            self.ema_values[product] = new_price
        else:
            self.ema_values[product] = (self.alpha * new_price) + ((1 - self.alpha) * self.ema_values[product])
        return self.ema_values[product]
    
    def predict_next_price(self, product: str):
        if len(self.price_history[product]) < 5:
            return None  # Not enough data to predict
        
        prices = np.array(list(self.price_history[product]))
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.reshape(-1, 1)
        self.model.fit(X, y)
        next_price = self.model.predict([[len(prices)]])[0][0]
        return next_price
    
    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def train_dqn(self):
        if len(self.memory) < 32:
            return  # Not enough experiences to train
        
        batch = random.sample(self.memory, 32)
        for state, action, reward, next_state in batch:
            target = reward
            if next_state is not None:
                target += self.discount_factor * np.max(self.dqn_model.predict(next_state.reshape(1, -1)))
            target_f = self.dqn_model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.dqn_model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
    
    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice([0, 1, 2])  # 0: Buy, 1: Sell, 2: Hold
        return np.argmax(self.dqn_model.predict(state.reshape(1, -1)))
    
    def run(self, state: TradingState):
        result = {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                fair_price = (best_bid + best_ask) / 2
                self.price_history[product].append(fair_price)
                ema = self.calculate_ema(product, fair_price)
                predicted_price = self.predict_next_price(product)
            else:
                fair_price = None
                predicted_price = None
            
            if fair_price:
                print(f"Trading {product}: Fair Price Estimated at {fair_price}, EMA: {ema}, Predicted Price: {predicted_price}")
                
                bid_vol = sum(order_depth.buy_orders.values())
                ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
                bid_ask_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) != 0 else 0
                
                state_vector = np.array([fair_price, bid_ask_imbalance, predicted_price or fair_price])
                action = self.select_action(state_vector)
                reward = 0
                
                if action == 0 and predicted_price and predicted_price > fair_price:
                    fair_price += 0.3  # Buy if price is expected to rise
                    reward = 1
                elif action == 1 and predicted_price and predicted_price < fair_price:
                    fair_price -= 0.3  # Sell if price is expected to fall
                    reward = 1
                else:
                    reward = -1  # No profitable action
                
                next_state_vector = np.array([fair_price, bid_ask_imbalance, predicted_price or fair_price])
                self.store_experience(state_vector, action, reward, next_state_vector)
                self.train_dqn()
                
                trade_qty = min(10, max(1, int(10 / (1 + abs(bid_ask_imbalance)))))
                
                for ask_price, ask_qty in order_depth.sell_orders.items():
                    if ask_price < fair_price:
                        orders.append(Order(product, ask_price, min(-ask_qty, trade_qty)))
                
                for bid_price, bid_qty in order_depth.buy_orders.items():
                    if bid_price > fair_price:
                        orders.append(Order(product, bid_price, -min(bid_qty, trade_qty)))
                
            result[product] = orders
        
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
        traderData = "Updated State"
        conversions = 0  # No conversions for now
        return result, conversions, traderData
