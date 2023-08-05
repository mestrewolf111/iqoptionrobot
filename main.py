import time
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
import requests
from keras.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from iqoptionapi.stable_api import IQ_Option
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
import os
tf.function(
	func=None,
	input_signature=None,
	autograph=True,
	jit_compile=None,
	reduce_retracing=True,
	experimental_implements=None,
	experimental_autograph_options=None,
	experimental_relax_shapes=True,
	experimental_compile=True,
	experimental_follow_type_hints=None
)
# tf.executing_eagerly()

tf.config.run_functions_eagerly(True)


@tf.function
def fn():
	with tf.init_scope():
		print(tf.executing_eagerly())
	print(tf.executing_eagerly())


fn()

try:
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
	# Memory growth must be set before GPUs have been initialized
	print(e)


warnings.filterwarnings("ignore")
class TradingBot:
	def __init__(self, email, password, par, time_frame, bet_money=100):
		self.iq = IQ_Option(email, password)
		self.iq.connect()
		self.par = par
		self.counter = 0
		self.wins = 0
		self.losses = 0
		self.ligado = "ONLINE"
		self.optimizer = Adam(learning_rate=0.01)
		self.reward = 1
		self.buffer_size = 1000
		self.status = 'ESPERANDO OPERACAO'
		self.direcao = 2
		self.X_buffer = []
		self.y_buffer = []
		self.num_epochs = 40
		self.time_frame = time_frame
		self.bet_money = bet_money
		self.model = None
		self.scaler = MinMaxScaler()
		self.action = None
		self.num_lags = 3
		self.X_train = None
		self.y_train = None

	def get_data_iq(self):
		velas = self.iq.get_candles(self.par, self.time_frame, 1000, time.time())
		data = pd.DataFrame(velas)
		return data

	def fit_model(self):
		# Converter os dados de treinamento para o formato correto
		X_train = np.array(self.X_train)
		y_train = np.array(self.y_train)
		# Treinar o modelo usando o otimizador e a função de perda personalizada
		self.model.compile(optimizer=self.optimizer, loss=self.custom_loss)
		self.model.fit(X_train, y_train, epochs=self.num_epochs, verbose=0)

	def trendlinaspercent(self,data):
		X = data
		data['LTA'] = data['close'].ewm(span=18, adjust=False).mean()
		data['LTB'] = data['close'].ewm(span=42, adjust=False).mean()
		# Calculate the difference between current price and LTA
		diff_lta = (data['close'] - data['LTA']).abs()
		# Calculate the difference between current price and LTB
		diff_ltb = (data['close'] - data['LTB']).abs()
		# Create a new column in the dataframe with 0 if the price is not near either LTA or LTB, 1 if near LTB and 2 if near LTA
		data['trendline_percentage'] = np.where(diff_lta < diff_ltb, 2, np.where(diff_lta > diff_ltb, 1, 0))
		return data['trendline_percentage']

	def checktempo(self):
		while True:
			time.sleep(1)
			if datetime.datetime.now().second == 2:
				break

	def checktempo2(self):
		while True:
			time.sleep(1)
			if datetime.datetime.now().second == 58:
				break

	def prepare_data(self, data):
		# Calculate price direction for each candle (1 for increase, 0 for decrease)
		data.loc[:, "price_direction"] = np.where(data["close"] > data["open"], 1, 0)
		# Remove unnecessary columns and shift the target variable up by one row
		data = data[["open", "close", "min", "max", "volume", "price_direction"]]
		data["price_direction"] = np.where(data['close'].shift(-1) < data['close'], 1,
		                        np.where(data['close'].shift(-1) > data['close'], 0,
		                                 0.5))
		# Drop the last row (NaN value due to shifting)
		data.dropna(inplace=True)
		return data




	def add_technical_indicators(self, data):
		rolling_window = data["close"].rolling(window=20)
		data["bollinger_mid"] = rolling_window.mean()
		data["bollinger_std"] = rolling_window.std()
		data["bollinger_upper"] = data["bollinger_mid"] + 1 * data["bollinger_std"]
		data["bollinger_lower"] = data["bollinger_mid"] - 1 * data["bollinger_std"]
		diff = data["close"].diff()
		gain = diff.where(diff > 0, 0)
		loss = -diff.where(diff < 0, 0)
		avg_gain = gain.rolling(window=14).mean()
		avg_loss = loss.rolling(window=14).mean()
		rs = avg_gain / avg_loss
		data["rsi"] = 100 - (100 / (1 + rs))
		# Calculate Stochastic RSI
		rsi_14 = data["rsi"]
		rsi_min_14 = rsi_14.rolling(window=14).min()
		rsi_max_14 = rsi_14.rolling(window=14).max()
		data["stochastic_rsi"] = (rsi_14 - rsi_min_14) / (rsi_max_14 - rsi_min_14)
		data.dropna(inplace=True)
		return data

	def add_lags(self, data, num_lags):
		for lag in range(1, num_lags+1):
			for col in ["open", "close", "min", "max", "volume"]:
				data.loc[:, f"{col}_lag{lag}"] = data[col].shift(lag)
		# Remove rows with NaN values caused by the lagging
		data.dropna(inplace=True)
		return data

	def custom_loss(self, action_probs, action_taken, rewards):
		# Converter os tensores para float32
		action_probs = tf.cast(action_probs, tf.float32)
		action_taken = tf.cast(action_taken, tf.float32)
		rewards = tf.cast(rewards, tf.float32)

		log_probs = tf.math.log(action_probs)
		log_probs_action = tf.reduce_sum(action_taken * log_probs, axis=1)
		loss = -tf.reduce_mean(log_probs_action * rewards)
		return loss

	def train_with_reinforcement(self, X_train, action_taken, reward_after_60s, reward):
		with tf.GradientTape() as tape:
			action_probs = self.model(X_train, training=True)
			loss = self.custom_loss(action_probs, action_taken, reward)
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

	def take_action(self, state):
		#model =  load_model("desisto2.h5")

		predicted_prob = self.model.predict(state)[0, 0]
		print(predicted_prob)
		predicted_prob =predicted_prob[-1]
		print(predicted_prob)
		if predicted_prob >= 0.52:
			action = 1  # Compra
		elif predicted_prob <= 0.48:
			action = 0  # Venda
		else:
			action = 2  # Espere
		return action, predicted_prob

	def check_reward(self, reward_after_60s):
		if reward_after_60s == 1:
			self.wins += 1
			print("Resultado após 60 segundos: GANHOU")
		elif reward_after_60s == -1:
			self.losses += 1
			print("Resultado após 60 segundos: PERDEU")
		else:
			print("Resultado após 60 segundos: INDEFINIDO")
		return reward_after_60s

	def initialize_training_data(self, num_features):
		self.X_train = np.empty((0, num_features))
		self.y_train = np.empty(0)

	def update_training_data(self, X_normalized, action):
		if self.X_train is None or self.y_train is None:
			num_features = X_normalized.shape[1]
			self.initialize_training_data(num_features)
		self.X_buffer.append(X_normalized)
		self.y_buffer.append(action)
		if len(self.X_buffer) > self.buffer_size:
			X_batch = np.array(self.X_buffer)
			y_batch = np.array(self.y_buffer)
			self.X_train = np.vstack((self.X_train, X_batch))
			self.y_train = np.append(self.y_train, y_batch)
			self.X_buffer = []
			self.y_buffer = []

	def getcoresLstm2(self,data):
		data = pd.DataFrame(data)
		X = data[["open", "close", "min", "max"]]
		data['pips'] = np.where(data['close'].shift(-1) < data['close'], 1,
		                        np.where(data['close'].shift(-1) > data['close'], 0,
		                                 0.5))
		return data['pips']

	def create_new_model(self, input_dim):
		model = Sequential()
		model.add(BatchNormalization())
		model.add(Dropout(0.2))
		model.add(LSTM(128, input_shape=(1, input_dim), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(64, return_sequences=True))
		model.add(Dropout(0.2))
		model.add(Dense(32, activation='tanh'))
		model.add(Dropout(0.2))
		model.add(Dense(32, activation='tanh'))
		model.add(Dropout(0.2))
		model.add(Dense(16, activation='tanh'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
		return model

	def train_model(self, X_train, y_train, num_epochs=40):
		model = self.create_new_model(input_dim=X_train.shape[2])
		checkpoint_callback = ModelCheckpoint("desisto2.h5", monitor='val_accuracy', verbose=1,
		                                      save_best_only=True, mode='max')
		model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=1,
		          validation_split=0.3, callbacks=[checkpoint_callback,EarlyStopping(monitor='accuracy', patience=8, verbose=1, mode='max'),
		                                           ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8,
                                                         verbose=1, mode='min')],)
		return model

	def real_time_trading(self):
		X_data = self.get_data_iq()
		data = self.prepare_data(X_data)
		data = self.add_technical_indicators(data)
		data = self.add_lags(data, self.num_lags)
		X = data.drop(columns=["price_direction"])
		y = self.getcoresLstm2(data)
		X = self.scaler.fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
		X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
		X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
		self.model = self.train_model(X_train, y_train)
		self.initialize_training_data(X.shape[1])
		while True:
			X_data = self.get_data_iq()
			data = self.prepare_data(X_data)
			data = self.add_technical_indicators(data)
			data = self.add_lags(data, self.num_lags)
			if len(data) > 0:
				X = data.drop(columns=["price_direction"])
				X_normalized = self.scaler.transform(X)
				X_normalized = np.expand_dims(X_normalized, axis=0)
				action, predicted_prob = self.take_action(X_normalized)

				status = 'ESPERANDO'
				self.checktempo2()
				print(predicted_prob, action)
				if action == 1:
					check, id = self.iq.buy(self.bet_money, self.par, "call", 1)
					print("Action: Compra")
					status = 'COMPRANDO'
				elif action == 0:
					check, id = self.iq.buy(self.bet_money, self.par, "put", 1)
					print("Ação: Venda")
					status = 'VENDENDO'
				else:
					print("Ação: Espere")
					status = 'ESPERANDO'
				self.action = action
				self.counter += 1
				# Verificar se a ação é diferente de "Espere" antes de prosseguir para a próxima iteração
				if action != 2:
					balance_before_entry = self.iq.get_balance()
					print(balance_before_entry)

					time.sleep(10)
					self.checktempo()
					# Coletar o saldo após a entrada
					balance_after_entry = self.iq.get_balance()
					# Verificar se o robô ganhou ou perdeu com base no saldo
					ver = balance_after_entry - balance_before_entry
					if ver > 1:
						reward_after_60s = 1
						print("Resultado após 60 segundos: GANHOU")
						self.rewards = 1
						self.check_reward(reward_after_60s)
						status = 'ESPERANDO OPERACAO'
					elif ver < 1:
						reward_after_60s = -1
						self.rewards = -1
						print("Resultado após 60 segundos: PERDEU")
						self.check_reward(reward_after_60s)
						status = 'ESPERANDO OPERACAO'
					self.update_training_data(X_normalized, action)
					print("Vitórias:", self.wins)
					print("Derrotas:", self.losses)
					print("reward:", self.rewards)
					self.train_with_reinforcement(X_normalized, action, reward_after_60s,self.rewards)
					checkpoint_dir = './logs'
					checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
					checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
					checkpoint.save(file_prefix=checkpoint_prefix)
					checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
					self.model.save("desisto2.h5")
					print("Modelo treinado salvo com sucesso!")
					self.counter += 1
					if self.counter >= 6:
						self.counter = 0
						self.model = self.train_model(X_train, y_train)
					else:
						print(self.counter)

				else:
					self.counter += 1
					if self.counter >= 6:
						self.counter = 0
						self.model = self.train_model(X_train, y_train)

				# Aguardar 5 segundos antes da próxima iteração
			time.sleep(5)

if __name__ == "__main__":
	email = "email"
	password = "senha"
	par = "EURUSD"
	time_frame = 60
	bot = TradingBot(email, password, par, time_frame)
	try:
		bot.model = load_model("desisto2.h5")
		print("Modelo carregado com sucesso!")
	except:
		bot.model = bot.create_new_model(input_dim=9)
		print("Novo modelo criado com sucesso!")
	# Iniciar o loop de produção
	bot.real_time_trading()

