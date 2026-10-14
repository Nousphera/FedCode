import os
import logging
import functools
import collections
import torch
import torchmetrics
import flwr as fl
import numpy as np
from sklearn.cluster import KMeans
from .codec import floating_weights, pack_calibration, unpack_calibration, is_calibration

'Run k-means on model weights.'
def apply_kmeans(weights, num_clusters=64):
	#from cuml.cluster import KMeans as KMeans # lazy import
	#kmeans = KMeans(n_clusters=num_clusters, init='scalable-k-means++', max_iter=300, n_init=10, random_state=0)
	kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
	kmeans.fit(weights)
	return (kmeans.cluster_centers_ , kmeans.labels_)

'Binary seach tree function.'
def binary_search(data, value):
	lo, hi = 0, len(data) - 1
	best_ind = lo
	while lo <= hi:
		mid = lo + (hi - lo) // 2
		if data[mid] < value:
			lo = mid + 1
		elif data[mid] > value:
			hi = mid - 1
		else:
			best_ind = mid
			break
		if abs(data[mid] - value) < abs(data[best_ind] - value):
			best_ind = mid
	return best_ind

'Update model state from aggregated clusters.'
def model_state_from_aggregated_clusters(model_state, clusters):
	centers = np.sort(np.asarray(clusters).reshape(-1))
	if not len(centers) or not np.isfinite(centers).all(): raise ValueError('Expected finite nonempty codebook')
	result = collections.OrderedDict()
	for name, param in model_state.items():
		param = np.asarray(param)
		if not np.issubdtype(param.dtype, np.floating):
			result[name] = param.copy()
			continue
		flat = param.reshape(-1)
		right = np.searchsorted(centers, flat).clip(0,len(centers)-1)
		left = (right-1).clip(0,len(centers)-1)
		choice = np.where(np.abs(flat-centers[left]) <= np.abs(flat-centers[right]), left, right)
		result[name] = centers[choice].reshape(param.shape).astype(param.dtype)
	return result


def compress_weights_with_codebook(model_state, clusters, labels, to_numpy=True, to_cpu=True):
	if to_numpy: model_state = {k:v.detach().cpu().numpy() for k,v in model_state.items()}
	return unpack_calibration(pack_calibration(model_state, clusters, labels))

class _FedAvg(fl.server.strategy.FedAvg):

	def __init__(self, init_state, num_clusters, *args, **kwargs):
		super(_FedAvg, self).__init__(*args, **kwargs)

		# New params
		self.model_state = init_state
		self.num_clusters = num_clusters
		self.initialized_clients = set()

	'FedAvg aggregation.'
	@staticmethod
	def aggregate(results): 
		# Calculate the total number of examples used during training
		num_examples_total = sum([num_examples for _, num_examples in results])
		# Create a list of weights, each multiplied by the related number of examples
		weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in results]
		# Compute average weights of each layer
		weights_prime = [functools.reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
		return weights_prime

	'Cluster aggregation.'
	@staticmethod
	def aggregate_clusters(results, model_state):
		aggregated_clusters = np.sort(np.concatenate([c for (c, _) in results], axis=0).flatten())
		model_state = model_state_from_aggregated_clusters(model_state=model_state, clusters=aggregated_clusters)
		weights = [val for _, val in model_state.items()]
		return (weights, model_state)

	def configure_fit(self, server_round, parameters, client_manager):
		config = dict(self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {})
		arrays = fl.common.parameters_to_ndarrays(parameters)
		state = collections.OrderedDict(zip(self.model_state.keys(), arrays))
		clusters, labels = apply_kmeans(floating_weights(state), self.num_clusters)
		calibration = pack_calibration(state, clusters, labels)
		sample_size, minimum = self.num_fit_clients(client_manager.num_available())
		clients = client_manager.sample(num_clients=sample_size, min_num_clients=minimum)
		instructions = []
		for client in clients:
			client_config = dict(config)
			# A late first participant has no local weight arrangement yet.
			cold_start = str(client.cid) not in self.initialized_clients
			client_config['broadcast_clusters'] = bool(config['broadcast_clusters'] and not cold_start)
			client_config['cold_start_calibration'] = cold_start
			payload = [np.sort(clusters.reshape(-1))] if client_config['broadcast_clusters'] else calibration
			instructions.append((client,fl.common.FitIns(fl.common.ndarrays_to_parameters(payload),client_config)))
		return instructions

	def aggregate_fit(self, server_round, results, failures,):

		if not results:
			return None, {}
		if not self.accept_failures and failures:
			return None, {}

		calibration_flags = {bool(res.metrics['send_compressed_weights']) for _,res in results}
		if len(calibration_flags) != 1: raise ValueError('Mixed uplink calibration modes in one round')
		received_compressed_weights = calibration_flags.pop()
		weights_results = []
		for proxy, res in results:
			payload = fl.common.parameters_to_ndarrays(res.parameters)
			weights = list(unpack_calibration(payload).values()) if received_compressed_weights else payload
			weights_results.append((weights, res.num_examples))
			self.initialized_clients.add(str(proxy.cid))
		if received_compressed_weights:
			parameters_aggregated = __class__.aggregate(weights_results)
			self.model_state = collections.OrderedDict((k,np.asarray(v,dtype=self.model_state[k].dtype))
				for k,v in zip(self.model_state,parameters_aggregated))
			parameters_aggregated = list(self.model_state.values())
		else:
			parameters_aggregated, self.model_state = __class__.aggregate_clusters(weights_results,self.model_state)

		parameters_aggregated = fl.common.ndarrays_to_parameters(parameters_aggregated)

		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

		return parameters_aggregated, metrics_aggregated

class Client(fl.client.NumPyClient):

	def __init__(self, cid, num_clients, model_loader, data_loader, split_fn=None,
			batch_size=128, seed=42, verbose=False, temp_dir='./tmp', device='cuda'):

		self.cid = int(cid)
		self.data, self.num_classes, self.num_samples =\
			data_loader(id=int(cid), num_shards=num_clients, split_fn=split_fn,
						batch_size=batch_size, seed=seed)
		self.model_loader = model_loader
		self.verbose = verbose
		self.device = device
		self.model_state_fp = os.path.join(temp_dir, f'client_{self.cid}_model_state.pth')

	def set_parameters(self, parameters, config):

		# Create model (if not available)
		if not hasattr(self, 'model'):
			self.model = self.model_loader(num_classes=self.num_classes).to(self.device)

		if config['broadcast_clusters']: # Convert to numpy and apply codebook
			# HACK: Load state from file - Normally this is stored on device.
			model_state = self.load_state_from_file()
			params_dict = model_state_from_aggregated_clusters(
				model_state={k: v.cpu().numpy() for k, v in model_state.items()}, clusters=parameters)
		else: # Extract parameters directly.
			params_dict = unpack_calibration(parameters) if is_calibration(parameters) else {k:v for k,v in zip(self.model.state_dict(),parameters)}

		# Load parameters
		state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict.items()})
		self.model.load_state_dict(state_dict, strict=True)

		# Metrics
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
		self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)

	def get_parameters(self, config=None):
		return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

	def encode_update(self, config):
		state = {k:v.detach().cpu().numpy() for k,v in self.model.state_dict().items()}
		clusters,labels = apply_kmeans(floating_weights(state),int(config['num_clusters']))
		compressed = compress_weights_with_codebook(state,clusters,labels,to_numpy=False)
		self.model.load_state_dict(collections.OrderedDict((k,torch.from_numpy(v)) for k,v in compressed.items()))
		self.store_state_to_file()
		if not config['request_compressed_weights']: return [clusters], 0
		return pack_calibration(state,clusters,labels), 1

	def fit(self, parameters, config):
		self.set_parameters(parameters, config)
		h = __class__.train(ds=self.data, model=self.model, epochs=config['epochs'],\
			optimizer=self.optimizer, metric=self.metric, verbose=self.verbose)
		weights, h['send_compressed_weights'] = self.encode_update(config=config)
		return weights, self.num_samples, h

	def evaluate(self, parameters, config):
		raise NotImplementedError('Client-side evaluation is not implemented!')

	def store_state_to_file(self):
		torch.save(self.model.state_dict(), f=self.model_state_fp)

	def load_state_from_file(self):
		return torch.load(self.model_state_fp, map_location=self.device, weights_only=True)

	@staticmethod
	def train(ds, model, epochs, optimizer, metric, verbose=False, logger=None):
		device = next(model.parameters()).device
		metric.reset()
		model.train()
		criterion = torch.nn.CrossEntropyLoss()
		score = []
		for epoch in range(epochs):
			metric.reset()
			train_loss = 0.0
			seen = 0
			for _, (x,y) in enumerate(ds):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
				optimizer.zero_grad()
				logits = model(x)
				#y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				_loss = criterion(logits, y) #torch.nn.functional.nll_loss(y_prob, y)
				_loss.backward()
				optimizer.step()
				train_loss += _loss.item() * y.numel()
				seen += y.numel()
				y_preds = torch.argmax(torch.nn.functional.log_softmax(logits, dim=1), dim=1)
				metric(y_preds, y)
			train_loss /= seen
			score.append(train_loss)
			acc = metric.compute()
			if verbose:
				if logger:
					logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")
				else:
					print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return {'loss': float(score[-1]), 'accuracy': float(acc.cpu().detach().numpy())}

class Server(fl.server.Server):

	def __init__(self, model_loader, data_loader, num_rounds, num_clients=10,
		participation=1.0, init_model=None, batch_size=128, device='cuda',
		num_epochs=4, lr=1e-3, num_clusters=64, start_rnd=2, broadcast_freq=5,
		request_freq=2, verbose=False, log_level=logging.INFO,
		file_logger=None, args_msg=''):

		self.num_rounds = num_rounds
		self.data, self.num_classes, self.num_samples = data_loader(batch_size=batch_size, return_eval_ds=True)
		self.model_loader = model_loader
		self.init_model = init_model
		self.clients_config = {"epochs":num_epochs, "lr":lr, "num_clusters":num_clusters}
		self.num_clients = num_clients
		self.participation = participation
		self.max_workers = None
		self.device = device
		self.verbose = verbose
		self._client_manager = fl.server.client_manager.SimpleClientManager()
		self.logger = logging.getLogger("flower")
		self.logger.setLevel(log_level)

		if file_logger is not None: self.logger.addHandler(logging.FileHandler(file_logger))
		self.logger.info(args_msg)

		# Extra parameters
		self.num_clusters = num_clusters
		self.set_strategy(self) # NOTE: After setting self.num_clusters!
		self.min_rnd = start_rnd
		self.request_freq = request_freq
		self.broadcast_freq = broadcast_freq
		self.broadcast_clusters_fn = lambda rnd,freq: rnd>self.min_rnd and (rnd%freq!=0)
		self.request_compressed_weights_from_clients_fn = lambda rnd,freq: rnd<self.min_rnd or (rnd%freq==0)

	def set_max_workers(self, *args, **kwargs):
		return super(Server, self).set_max_workers(*args, **kwargs)

	def set_strategy(self, *_):
		init_weights = self.get_initial_parameters()
		self.strategy = _FedAvg(
			min_available_clients=self.num_clients, fraction_fit=self.participation,
			min_fit_clients=max(1,int(self.participation*self.num_clients)), fraction_evaluate=0.0,
			min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
			on_fit_config_fn=self.get_client_config_fn(), initial_parameters=init_weights,
			init_state=self.init_state, num_clusters=self.num_clusters)

	def client_manager(self, *args, **kwargs):
		return super(Server, self).client_manager(*args, **kwargs)

	def get_parameters(self, config={}):
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

	def set_parameters(self, parameters, config):

		if not hasattr(self, 'model'):
			self.model = self.model_loader(num_classes=self.num_classes).to(self.device)

		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)
		self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)

	def get_initial_parameters(self, *_):
		if self.init_model is not None:
			self.init_state = {k:v.cpu().numpy() for k, v in \
				torch.load(self.init_model, map_location=self.device, weights_only=True).items()}
		else:
			self.init_state = {k : v.cpu().numpy() for k, v in \
				self.model_loader(num_classes=self.num_classes).state_dict().items()}
		return fl.common.ndarrays_to_parameters([v for _,v in self.init_state.items()])

	def get_evaluation_fn(self):
		def evaluation_fn(rnd, parameters, config):
			self.set_parameters(parameters, config)
			metrics = __class__.evaluate(model=self.model, ds=self.data, metric=self.metric, verbose=self.verbose, logger=self.logger)
			return metrics[0], {"accuracy":metrics[1]}
		return evaluation_fn

	def get_client_config_fn(self):
		def get_on_fit_config_fn(rnd):
			self.clients_config["rnd"] = rnd
			self.clients_config['broadcast_clusters'] = self.broadcast_clusters_fn(rnd,self.broadcast_freq)
			self.clients_config['request_compressed_weights'] = self.request_compressed_weights_from_clients_fn(rnd,self.request_freq)
			msg = f"[Server] - Round {self.clients_config['rnd']}: Server Broadcast " + \
				('Compressed Weights' if not self.clients_config['broadcast_clusters'] else 'Clusters') + ' | Clients Broadcast ' + \
				('Compressed Weights' if self.clients_config['request_compressed_weights'] else 'Clusters') + '.'
			print(msg)
			return self.clients_config
		return get_on_fit_config_fn

	@staticmethod
	def evaluate(ds, model, metric, verbose=False, logger=None):
		device = next(model.parameters()).device
		metric.reset()
		model.eval()
		criterion = torch.nn.CrossEntropyLoss()
		test_loss = 0.0
		seen = 0
		with torch.no_grad():
			for _, (x, y) in enumerate(ds):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
				logits = model(x)
				#y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				test_loss += criterion(logits, y).item() * y.numel()
				seen += y.numel() # #torch.nn.functional.nll_loss(y_prob, y).item()
				y_preds = torch.argmax(torch.nn.functional.log_softmax(logits, dim=1), dim=1)
				metric(y_preds, y)
		test_loss /= seen
		acc = metric.compute()
		if verbose:
			if logger is not None:
				logger.info(f"Loss: {test_loss:.4f} - Accuracy: {100. * acc:.2f}%")
			else:
				print(f"Loss: {test_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return (test_loss, float(acc.cpu().detach().numpy()))

