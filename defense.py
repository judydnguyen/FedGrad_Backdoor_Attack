import pdb
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from utils import *

from geometric_median import geometric_median
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics.pairwise as smp
import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
import hdbscan
from sklearn.cluster import KMeans


# from all_utils import calculate_sum_grad_diff
# import logging
# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()

def extract_classifier_layer(net_list, global_avg_net, prev_net, model="vgg9"):
    bias_list = []
    weight_list = []
    weight_update = []
    avg_bias = None
    avg_weight = None
    prev_avg_bias = None
    prev_avg_weight = None
    last_model_layer = "classifier" if model=="vgg9" else "fc3" 
    
    if model == "vgg9":
        for idx, param in enumerate(global_avg_net.classifier.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.classifier.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.classifier.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
            
    elif model == "lenet":
        for idx, param in enumerate(global_avg_net.fc2.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()
        for idx, param in enumerate(prev_net.fc2.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.fc2.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
    
    return bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight

def extract_last_layer(net, model="vgg9"):
    bias, weight = None, None
    if model == "vgg9":
        for idx, param in enumerate(net.classifier.parameters()):
            if idx:
                bias = param.data.cpu().numpy()
            else:
                weight = param.data.cpu().numpy()
    elif model == "lenet":
        for idx, param in enumerate(net.fc2.parameters()):
            if idx:
                bias = param.data.cpu().numpy()
            else:
                weight = param.data.cpu().numpy()
    return bias, weight   
 
def rlr_avg(vectorize_nets, vectorize_avg_net, freq, attacker_idxs, lr, n_params, device, robustLR_threshold=4):
    lr_vector = torch.Tensor([lr]*n_params).to(device)
    total_client = len(vectorize_nets)
    local_updates = vectorize_nets - vectorize_avg_net
    print(f"len freq: {len(freq)}")
    print(f"local_updates.shape is: {len(local_updates)}")
    fed_avg_updates_vector = np.average(local_updates, weights=freq, axis=0).astype(float32)
    print(f"fed_avg_vector.shape is: {fed_avg_updates_vector.shape}")
    # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
    selected_net_indx = [i for i in range(total_client) if i not in attacker_idxs]
    selected_freq = np.array(freq)[selected_net_indx]
    selected_freq = [freq/sum(selected_freq) for freq in selected_freq]
    
    
    agent_updates_sign = [np.sign(update) for update in local_updates]  
    sm_of_signs = np.abs(sum(agent_updates_sign))
    sm_of_signs[sm_of_signs < robustLR_threshold] = -lr
    sm_of_signs[sm_of_signs >= robustLR_threshold] = lr
    print(f"sm_of_signs is: {sm_of_signs}")
    
    lr_vector = sm_of_signs
    poison_w_idxs = sm_of_signs < 0
    # poison_w_idxs = poison_w_idxs*1
    print(f"poison_w_idxs: {poison_w_idxs}")
    print(f"lr_vector: {lr_vector}")
    local_updates = np.asarray(local_updates)
    print(f"local_updates.shape is: {local_updates.shape}")
    # local_updates[attacker_idxs][poison_w_idxs] = 0
    cnt = 0
    sm_updates_2 = 0
    # for _id, update in enumerate(local_updates):
    #     if _id not in attacker_idxs:
    #         sm_updates_2 += selected_freq[cnt]*update[poison_w_idxs]
    #         cnt+=1
    for _id, update in enumerate(local_updates):
        if _id not in attacker_idxs:
            sm_updates_2 += freq[_id]*update[poison_w_idxs]
        else:
            sm_updates_2 += freq[_id]*(-update[poison_w_idxs])
            
    print(f"sm_updates_2.shape is: {sm_updates_2.shape}")
    fed_avg_updates_vector[poison_w_idxs] = sm_updates_2
    new_global_params =  (vectorize_avg_net + lr*fed_avg_updates_vector).astype(np.float32)
    return new_global_params

class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


# class ClippingDefense(Defense):
#     """
#     Deprecated, do not use this method
#     """
#     def __init__(self, norm_bound, *args, **kwargs):
#         self.norm_bound = norm_bound

#     def exec(self, client_model, *args, **kwargs):
#         vectorized_net = vectorize_net(client_model)
#         weight_norm = torch.norm(vectorized_net).item()
#         clipped_weight = vectorized_net/max(1, weight_norm/self.norm_bound)

#         logger.info("Norm Clipped Mode {}".format(
#             torch.norm(clipped_weight).item()))
#         load_model_weight(client_model, clipped_weight)        
#         # index_bias = 0
#         # for p_index, p in enumerate(client_model.parameters()):
#         #     p.data =  clipped_weight[index_bias:index_bias+p.numel()].view(p.size())
#         #     index_bias += p.numel()
#         ##weight_norm = torch.sqrt(sum([torch.norm(p)**2 for p in client_model.parameters()]))
#         #for p_index, p in enumerate(client_model.parameters()):
#         #    p.data /= max(1, weight_norm/self.norm_bound)
#         return None

# CAN REMOVE NON-EXPERIMENTED CLASSES

# class WeightDiffClippingDefense(Defense):
#     def __init__(self, norm_bound, *args, **kwargs):
#         self.norm_bound = norm_bound

#     def exec(self, client_model, global_model, *args, **kwargs):
#         """
#         global_model: the global model at iteration T, bcast from the PS
#         client_model: starting from `global_model`, the model on the clients after local retraining
#         """
#         vectorized_client_net = vectorize_net(client_model)
#         vectorized_global_net = vectorize_net(global_model)
#         vectorize_diff = vectorized_client_net - vectorized_global_net

#         weight_diff_norm = torch.norm(vectorize_diff).item()
#         clipped_weight_diff = vectorize_diff/max(1, weight_diff_norm/self.norm_bound)

#         logger.info("Norm Weight Diff: {}, Norm Clipped Weight Diff {}".format(weight_diff_norm,
#             torch.norm(clipped_weight_diff).item()))
#         load_model_weight_diff(client_model, clipped_weight_diff, global_model)
#         return None

# class WeakDPDefense(Defense):
#     """
#         deprecated: don't use!
#         according to literature, DPDefense should be applied
#         to the aggregated model, not invidual models
#         """
#     def __init__(self, norm_bound, *args, **kwargs):
#         self.norm_bound = norm_bound

#     def exec(self, client_model, device, *args, **kwargs):
#         self.device = device
#         vectorized_net = vectorize_net(client_model)
#         weight_norm = torch.norm(vectorized_net).item()
#         clipped_weight = vectorized_net/max(1, weight_norm/self.norm_bound)
#         dp_weight = clipped_weight + torch.randn(
#             vectorized_net.size(),device=self.device) * self.stddev

#         load_model_weight(client_model, clipped_weight)
#         return None

# class AddNoise(Defense):
#     def __init__(self, stddev, *args, **kwargs):
#         self.stddev = stddev

#     def exec(self, client_model, device, *args, **kwargs):
#         self.device = device
#         vectorized_net = vectorize_net(client_model)
#         gaussian_noise = torch.randn(vectorized_net.size(),
#                             device=self.device) * self.stddev
#         dp_weight = vectorized_net + gaussian_noise
#         load_model_weight(client_model, dp_weight)
#         logger.info("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))
        
#         return None


class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps, g_user_indices, device, *args, **kwargs):
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        
        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
            neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]
            
            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))
            #aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0)
            aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq, axis=0).astype(np.float32)

            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
            neo_net_list = [aggregated_model]
            #logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq

class RFA(Defense):
    """
    we implement the robust aggregator at: 
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation: 
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq, 
                   maxiter=4, eps=1e-5,
                   ftol=1e-6, device=torch.device("cuda"), 
                    *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        logger.info('Starting Weiszfeld algorithm')
        logger.info(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            logger.info("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        #logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        #tot_weights = np.sum(weights)
        #weighted_updates = [np.zeros_like(v) for v in points[0]]
        #for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        #return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # this is a helper function
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])
class GeoMedian(Defense):
    """
    we implement the robust aggregator of Geometric Median (GM)
    """
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq, 
                   maxiter=4, eps=1e-5,
                   ftol=1e-6, device=torch.device("cuda"), 
                    *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = np.array([vectorize_net(cm).detach().cpu().numpy() for cm in client_models]).astype(np.float32)
        median = geometric_median(vectorize_nets)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

# class CONTRA(Defense):

#     """
#     REIMPLEMENT OF CONTRA ALGORITHM
#     Awan, S., Luo, B., Li, F. (2021). 
#     CONTRA: Defending Against Poisoning Attacks in Federated Learning.
#     In: Bertino, E., Shulman, H., Waidner, M. (eds) 
#     Computer Security – ESORICS 2021. 
#     ESORICS 2021. Lecture Notes in Computer Science(), 
#     vol 12972. Springer, Cham. https://doi.org/10.1007/978-3-030-88418-5_22
#     """
#     def __init__(self, *args, **kwargs):
#         pass
    
#     def exec(self, client_models, net_freq, selected_node_indices, historical_local_updates, reputations, delta, threshold, k = 3, device=torch.device("cuda"), *args, **kwargs):
        
#         vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
#         training_client_cnt = len(selected_node_indices)
#         total_clients = len(historical_local_updates)
#         avg_k_top_cs = [0.0 for _ in range(total_clients)]
#         pairwise_cs = np.zeros((total_clients, total_clients))
#         # pairwise_cs = np.zeros((training_client_cnt, training_client_cnt))


#         for net_idx, global_node_id in enumerate(selected_node_indices):
#             i_local_updates = np.asarray(historical_local_updates[global_node_id])
#             cs_i = []
#             for net_idx_p, global_node_id_p in enumerate(selected_node_indices):
#                 if global_node_id_p != global_node_id:
#                     p_update = historical_local_updates[global_node_id_p]
#                     if len(p_update) > 1:
#                         cs_p_i = np.dot(i_local_updates, np.asarray(historical_local_updates[global_node_id_p]))/(np.linalg.norm(i_local_updates)*np.linalg.norm(historical_local_updates[global_node_id_p]))
#                     else:
#                         cs_p_i = 0.0
#                     cs_i.append(cs_p_i)
#                     pairwise_cs[global_node_id][global_node_id_p] = cs_p_i
                        
#             # for client_p in range(total_clients):
#             #     if client_p+1 != global_node_id:
#             #         p_update = historical_local_updates[client_p]
#             #         if len(p_update) > 1:
#             #             cs_p_i = np.dot(i_local_updates, np.asarray(historical_local_updates[client_p]))/(np.linalg.norm(i_local_updates)*np.linalg.norm(historical_local_updates[client_p]))
#             #         else:
#             #             cs_p_i = 0.0
#             #         cs_i.append(cs_p_i)
#             #         pairwise_cs[global_node_id][client_p] = cs_p_i
            
#             cs_i = np.asarray(cs_i)
#             cs_i[::-1].sort()
#             avg_k_top_cs_i = np.average(cs_i[:k])
#             if avg_k_top_cs_i > threshold:
#                 reputations[global_node_id] -= delta
#             else:
#                 reputations[global_node_id] += delta
#             avg_k_top_cs[global_node_id] = avg_k_top_cs_i

#         alignment_levels = pairwise_cs.copy()
#         lr_list = [1.0 for _ in range(total_clients)]
#         # for net_idx, global_node_id in enumerate(selected_node_indices):
#         #     lr_list[global_node_id] = 1.0
#         # print("avg_k_top_cs: ", avg_k_top_cs)
#         for net_idx in selected_node_indices:
#             cs_m_n = []
#             for client_p in selected_node_indices:
#                 if client_p != net_idx: 
#                     if avg_k_top_cs[client_p] > avg_k_top_cs[net_idx]:
#                         alignment_levels[net_idx][client_p] *= float(avg_k_top_cs[net_idx]/avg_k_top_cs[client_p])
#                     # else:
#                     #     alignment_levels[net_idx][client_p] *= min(1.0, float(avg_k_top_cs[net_idx])/1.0)
#             a = np.asarray(alignment_levels[net_idx])
#             a = a[a != 0.0]
#             # print("a: ", a)
#             # print(max(a))
#             # print("alignment_levels[net_idx]: ", alignment_levels[net_idx])
#             # print("max(alignment_levels[net_idx]: ", max(alignment_levels[net_idx]))
#             # print(alignment_levels[net_idx].shape)
#             lr_net_idx = 1.0 - np.amax(alignment_levels[net_idx])
#             #print("lr_net_idx: ", lr_net_idx)
#             lr_list[net_idx] = lr_net_idx
#             reputations[net_idx] = max(np.asarray(reputations))
#         #print("alignment_levels: ", alignment_levels)
#         lr_final = []
#         for net_idx, global_node_id in enumerate(selected_node_indices):
#             lr_final.append(lr_list[global_node_id])
#         #print("lr_list first: ", lr_final)
#         lr_list = np.asarray(lr_final)
#         lr_list = lr_list/(max(lr_list))
#         print("lr_list: ", lr_list)
#         for i, lr in enumerate(lr_list):
#             if(lr == 1.0):
#                 lr_list[i] = logit(0.99)+0.5
#             else:
#                 lr_list[i] = logit(lr_list[i]) + 0.5
#         # lr_list = logit(lr_list/(1.0-lr_list)) + 0.5 
#         print("lr_list: ", lr_list)
#         weights = lr_list.copy()
#         print("weights: ", weights)
#         aggregated_w = self.weighted_average_oracle(vectorize_nets, weights)
#         aggregated_model = client_models[0] # slicing which doesn't really matter
#         load_model_weight(aggregated_model, torch.from_numpy(aggregated_w.astype(np.float32)).to(device))
#         neo_net_list = [aggregated_model]
#         neo_net_freq = [1.0]
#         return neo_net_list, neo_net_freq, reputations
        
#     def weighted_average_oracle(self, points, weights):
#         """Computes weighted average of atoms with specified weights
#         Args:
#             points: list, whose weighted average we wish to calculate
#                 Each element is a list_of_np.ndarray
#             weights: list of weights of the same length as atoms
#         """
#         ### original implementation in TFF
#         #tot_weights = np.sum(weights)
#         #weighted_updates = [np.zeros_like(v) for v in points[0]]
#         #for w, p in zip(weights, points):
#         #    for j, weighted_val in enumerate(weighted_updates):
#         #        weighted_val += (w / tot_weights) * p[j]
#         #return weighted_updates
#         ####
#         tot_weights = np.sum(weights)
#         weighted_updates = np.zeros(points[0].shape)
#         for w, p in zip(weights, points):
#             weighted_updates += (w * p / tot_weights)
#         return weighted_updates
    
class FedGrad(Defense):
    """
    FedGrad by DungNT
    """
    def __init__(self, total_workers, num_workers, num_adv, num_valid = 1, instance="benchmark", use_trustworthy=False, *args, **kwargs):
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv
        self.instance = instance
        self.choosing_frequencies = {}
        self.accumulate_c_scores = {}
        self.use_trustworthy = use_trustworthy
        self.pairwise_w = np.zeros((total_workers+1, total_workers+1))
        self.pairwise_b = np.zeros((total_workers+1, total_workers+1))
        self.eta = 0.5 # this parameter could be changed
        self.switch_round = 50 # this parameter could be changed
        self.trustworthy_threshold = 0.75
        self.lambda_1 = 0.25
        self.lambda_2 = 1.0
        
        logger.info("Starting performing FedGrad...")
        self.pairwise_choosing_frequencies = np.zeros((total_workers, total_workers))
        self.trustworthy_scores = [[0.5] for _ in range(total_workers+1)]

    def exec(self, client_models, num_dps, net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, model_name, device, *args, **kwargs):
        start_fedgrad_t = time.time()*1000
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        neighbor_distances = []
        logger.info("Starting performing FedGrad...")
        
        # SOFT FILTER
        layer1_start_t = time.time()*1000
        bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight = extract_classifier_layer(client_models, pseudo_avg_net, net_avg, model_name)
        total_client = len(g_user_indices)
        
        raw_c_scores = self.get_compromising_scores(glob_update, weight_update)
        c_scores = []
        for idx, cli in enumerate(g_user_indices):
            # increase the frequency of the selected choosen clients
            self.choosing_frequencies[cli] = self.choosing_frequencies.get(cli, 0) + 1
            # update the accumulator
            self.accumulate_c_scores[cli] = ((self.choosing_frequencies[cli] - 1) / self.choosing_frequencies[cli]) * self.accumulate_c_scores.get(cli, 0) + (1 / self.choosing_frequencies[cli]) *  raw_c_scores[idx]
            c_scores.append(self.accumulate_c_scores[cli])
        
        c_scores = np.array(c_scores)
        epsilon_1 = min(self.eta, np.median(c_scores))
        
        
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        
        suspicious_idxs_1 = [ind_ for ind_ in range(total_client) if c_scores[ind_] > epsilon_1]
        print("[Soft-filter] predicted suspicious set is:: ", suspicious_idxs_1)
        layer1_end_t = time.time()*1000
        layer1_inf_time = layer1_end_t-layer1_start_t
        print(f"Total computation time of the 1st layer is: {layer1_inf_time}")
        
        # HARD FILTER
        layer2_start_t = time.time()*1000
        round_pw_bias = np.zeros((total_client, total_client))
        round_pw_weight = np.zeros((total_client, total_client))
        
        sum_diff_by_label, glob_temp_sum_by_label = calculate_sum_grad_diff(meta_data = weight_update, num_w = weight_update[0].shape[-1], glob_update=glob_update)
        norm_bias_list = normalize(bias_list, axis=1)
        norm_grad_diff_list = normalize(sum_diff_by_label, axis=1)
        
        # UPDATE CUMULATIVE COSINE SIMILARITY 
        for i, g_i in enumerate(g_user_indices):
            distance = []
            for j, g_j in enumerate(g_user_indices):
                self.pairwise_choosing_frequencies[g_i][g_j] = self.pairwise_choosing_frequencies[g_i][g_j] + 1.0
                bias_p_i = norm_bias_list[i]
                bias_p_j = norm_bias_list[j]
                cs_1 = np.dot(bias_p_i, bias_p_j)/(np.linalg.norm(bias_p_i)*np.linalg.norm(bias_p_j))
                round_pw_bias[i][j] = cs_1.flatten()
                
                w_p_i = norm_grad_diff_list[i]
                w_p_j = norm_grad_diff_list[j]
                cs_2 = np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
                round_pw_weight[i][j] = cs_2.flatten()
       
        # compute closeness scores 
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)

        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        trusted_index = scores.index(min(scores)) # ==> trusted client is the client whose smallest closeness score.

        scaler = MinMaxScaler()
        round_pw_bias = scaler.fit_transform(round_pw_bias)
        round_pw_weight = scaler.fit_transform(round_pw_weight)

        # update cumulative information
        for i, g_i in enumerate(g_user_indices):
            for j, g_j in enumerate(g_user_indices):
                freq_appear = self.pairwise_choosing_frequencies[g_i][g_j]
                self.pairwise_w[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_w[g_i][g_j] +  1/freq_appear*round_pw_weight[i][j]
                self.pairwise_b[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_b[g_i][g_j] +  1/freq_appear*round_pw_bias[i][j]
                
        
        # From now on, trusted_model contains the index base model treated as valid user.
        suspicious_idxs_2 = []
        saved_pairwise_sim = []
        layer2_inf_t = 0.0
        
        final_suspicious_idxs = suspicious_idxs_1 # temporarily assigned by the first filter
        # NOW CHECK FOR SWITCH ROUND
        # TODO: find dynamic threshold
        # STILL PERFORM HARD-FILTER to save the historical information about colluding property.
        cummulative_w = self.pairwise_w[np.ix_(g_user_indices, g_user_indices)]
        cummulative_b = self.pairwise_b[np.ix_(g_user_indices, g_user_indices)]
        
        saved_pairwise_sim = np.hstack((cummulative_w, cummulative_b))
        kmeans = KMeans(n_clusters = 2)
        pred_labels = kmeans.fit_predict(saved_pairwise_sim)
        trusted_cluster_idx = pred_labels[trusted_index] # assign cluster containing trusted client as benign cluster
        malicious_cluster_idx = 0 if trusted_cluster_idx == 1 else 1
        suspicious_idxs_2 = np.argwhere(np.asarray(pred_labels) == malicious_cluster_idx).flatten()
        
        print("[Hard-filter] predicted suspicious set is: ", suspicious_idxs_2)
        layer2_end_t = time.time()*1000
        layer2_inf_t = layer2_end_t-layer2_start_t
        print(f"Total computation time of the 2nd layer is: {layer2_inf_t}")
        pseudo_final_suspicious_idxs = np.union1d(suspicious_idxs_2, suspicious_idxs_1).flatten()

        if round >= self.switch_round:
            final_suspicious_idxs = pseudo_final_suspicious_idxs
        print(f"[Combination-result] predicted suspicious set is: {final_suspicious_idxs}")

        # STARTING USING TRUSTWORTHY SCORES
        filtered_suspicious_idxs = list(final_suspicious_idxs.copy())
        if round >= self.switch_round:
            # for idx in final_suspicious_idxs:
            #     g_idx = g_user_indices[idx]
            #     if np.average(self.trustworthy_scores[g_idx]) >= self.trustworthy_threshold:
            #         filtered_suspicious_idxs.remove(idx)
            filtered_suspicious_idxs = [idx for idx in final_suspicious_idxs if np.average(self.trustworthy_scores[g_user_indices[idx]]) < self.trustworthy_threshold]
       
        if not filtered_suspicious_idxs:
            filtered_suspicious_idxs = suspicious_idxs_1 
        print(f"[Filtered-result] predicted suspicious set is: {filtered_suspicious_idxs}")     
                 
        if self.use_trustworthy: # used for ablation study
            final_suspicious_idxs = filtered_suspicious_idxs
        print(f"[Final-result] predicted suspicious set is: {final_suspicious_idxs}")   

        for idx, g_idx in enumerate(g_user_indices):
            if idx in final_suspicious_idxs:
                self.trustworthy_scores[g_idx].append(self.lambda_1)
            else:
                self.trustworthy_scores[g_idx].append(self.lambda_2)
        
        #GET ADDITIONAL INFORMATION of TPR and FPR, TNR
        tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 0.0, 0.0, 0.0
        tp_fedgrad_pred = []
        for id_ in participated_attackers:
            tp_fedgrad_pred.append(1.0 if id_ in final_suspicious_idxs else 0.0)
        fp_fegrad = len(final_suspicious_idxs) - sum(tp_fedgrad_pred)
        
        # Calculate true positive rate (TPR = TP/(TP+FN))
        total_positive = len(participated_attackers)
        total_negative = total_client - total_positive
        tpr_fedgrad = 1.0
        if total_positive > 0.0:
            tpr_fedgrad = sum(tp_fedgrad_pred)/total_positive
        # False postive rate
        fpr_fedgrad = fp_fegrad/total_negative
        tnr_fedgrad = 1.0 - fpr_fedgrad
        
        end_fedgrad_t = time.time()*1000
        fedgrad_t = end_fedgrad_t - start_fedgrad_t # finish calculating the computation time of FedGrad
        
        # tpr_fedgrad, fpr_fedgrad, tnr_fedgrad = 1.0, 1.0, 1.0
        neo_net_list = []
        neo_net_freq = []
        selected_net_indx = []
        for idx, net in enumerate(client_models):
            if idx not in final_suspicious_idxs:
                neo_net_list.append(net)
                neo_net_freq.append(1.0)
                selected_net_indx.append(idx)
        if len(neo_net_list) == 0:
            return [net_avg], [1.0], [], tpr_fedgrad, fpr_fedgrad, tnr_fedgrad, layer1_inf_time, layer2_inf_t, fedgrad_t
            
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
        selected_num_dps = np.array(num_dps)[selected_net_indx]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        logger.info("The chosen ones are users: {}, which are global users: {}".format(selected_net_indx, [g_user_indices[ti] for ti in selected_net_indx]))
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        pred_g_attacker = [g_user_indices[i] for i in final_suspicious_idxs]
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, pred_g_attacker, tpr_fedgrad, fpr_fedgrad, tnr_fedgrad, layer1_inf_time, layer2_inf_t, fedgrad_t

    def get_compromising_scores(self, global_update, weight_update):
        cs_dist = get_cs_on_base_net(weight_update, global_update)
        score = np.array(cs_dist)
        norm_score = min_max_scale(score)
        return norm_score

class RLR(Defense):
    def __init__(self, n_params, device, args, agent_data_sizes=[], writer=None, robustLR_threshold = 0, aggr="avg", poisoned_val_loader=None):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.n_params = n_params
        self.poisoned_val_loader = None
        self.cum_net_mov = 0
        self.device = device
        self.robustLR_threshold = robustLR_threshold
        
         
    def exec(self, global_model, client_models, num_dps, agent_updates_dict=None, cur_round=0):
        # adjust LR if robust LR is selected
        print(f"self.args: {self.args}")
        print(f"self.args['server_lr']: {self.args['server_lr']}")
        lr_vector = torch.Tensor([self.args['server_lr']]*self.n_params).to(self.device)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(global_model).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net
    
        if self.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(local_updates)
        
        
        aggregated_updates = 0
        if self.args['aggr']=='avg':          
            aggregated_updates = self.agg_avg(local_updates, num_dps)
        elif self.args['aggr']=='comed':
            #TODO update for the 2 remaining func
            aggregated_updates = self.agg_comed(local_updates)
        elif self.args['aggr'] == 'sign':
            aggregated_updates = self.agg_sign(local_updates)
            
        if self.args['noise'] > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args['noise']*self.args['clip'], size=(self.n_params,)).to(self.device))

        cur_global_params = vectorize_avg_net
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).astype(np.float32)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(new_global_params).to(self.device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq
     
    
    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [np.sign(update) for update in agent_updates]  
        sm_of_signs = np.abs(sum(agent_updates_sign))
        print(f"sm_of_signs is: {sm_of_signs}")
        
        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.args['server_lr']
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.args['server_lr']                                            
        return sm_of_signs
        
            
    def agg_avg(self, agent_updates_dict, num_dps):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in enumerate(agent_updates_dict):
            n_agent_data = num_dps[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args['clip']))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return
        
    # def comp_diag_fisher(self, model_params, data_loader, adv=True):

    #     model = models.get_model(self.args.data)
    #     vector_to_parameters(model_params, model.parameters())
    #     params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    #     precision_matrices = {}
    #     for n, p in deepcopy(params).items():
    #         p.data.zero_()
    #         precision_matrices[n] = p.data
            
    #     model.eval()
    #     for _, (inputs, labels) in enumerate(data_loader):
    #         model.zero_grad()
    #         inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
    #                                 labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
    #         if not adv:
    #             labels.fill_(self.args.base_class)
                
    #         outputs = model(inputs)
    #         log_all_probs = F.log_softmax(outputs, dim=1)
    #         target_log_probs = outputs.gather(1, labels)
    #         batch_target_log_probs = target_log_probs.sum()
    #         batch_target_log_probs.backward()
            
    #         for n, p in model.named_parameters():
    #             precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
                
    #     return parameters_to_vector(precision_matrices.values()).detach()

        
    # def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
    #     """ Getting sign agreement of updates between honest and corrupt agents """
    #     # total update for this round
    #     update = new_global_params - cur_global_params
        
    #     # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
    #     fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
    #     fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
    #     _, adv_idxs = fisher_adv.sort()
    #     _, hon_idxs = fisher_hon.sort()
        
    #     # get most important n_idxs params
    #     n_idxs = self.args.top_frac #math.floor(self.n_params*self.args.top_frac)
    #     adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
    #     hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()
        
    #     # minimized and maximized indexes
    #     min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
    #     max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()
        
    #     # get minimized and maximized idxs for adversary and honest
    #     max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
    #     max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
    #     min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
    #     min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)
       
    #     # get differences
    #     max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
    #     max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
    #     min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
    #     min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)
        
    #     # get actual update values and compute L2 norm
    #     max_adv_only_upd = update[max_adv_only_idxs] # S1
    #     max_hon_only_upd = update[max_hon_only_idxs] # S2
        
    #     min_adv_only_upd = update[min_adv_only_idxs] # S3
    #     min_hon_only_upd = update[min_hon_only_idxs] # S4


    #     #log l2 of updates
    #     max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
    #     max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
    #     min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
    #     min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()
       
    #     self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
    #     self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
    #     self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
    #     self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)
        
        
    #     net_adv =  max_adv_only_upd_l2 - min_adv_only_upd_l2
    #     net_hon =  max_hon_only_upd_l2 - min_hon_only_upd_l2
    #     self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
    #     self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)
        
    #     self.cum_net_mov += (net_hon - net_adv)
    #     self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
    #     return

class FLAME(Defense):
    """ We re-implement FLAME defense based on the original paperwork 
    and the pseudo-code provided by the authors at 
    https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    def exec(self, client_models, net_avg, device, *args, **kwargs):
        total_client = len(client_models)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(net_avg).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net
        
        #FILTERING C1:
        pairwise_cs = np.zeros((total_client, total_client))
        for i, w_p_i in enumerate(vectorize_nets):
            for j, w_p_j in enumerate(vectorize_nets):
                pairwise_cs[i][j] = 1.0 - np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
        pairwise_cs = normalize(pairwise_cs)
        
        min_cluster_sz = int(total_client/2+1)
        hb_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_sz, min_samples=1)
        hb_clusterer.fit(pairwise_cs)
        layer_1_pred_labels = hb_clusterer.labels_
        layer_1_pred_labels = np.asarray(layer_1_pred_labels)

        values, counts = np.unique(layer_1_pred_labels, return_counts=True)

        normal_client_label = layer_1_pred_labels[np.argmax(counts)]
        normal_client_idxs = np.argwhere(layer_1_pred_labels == normal_client_label).flatten()
        eucl_dist = []
        for i, g_p_i in enumerate(vectorize_nets):
            ds = g_p_i-vectorize_avg_net
            el_dis = np.sqrt(np.dot(ds, ds.T)).flatten()
            eucl_dist.append(el_dis)
        s_t = np.median(eucl_dist)
        
        normal_w = []
        for _id in normal_client_idxs:
            dym_thres = s_t/eucl_dist[_id]
            w_c = vectorize_avg_net + local_updates[_id]*min(1.0, dym_thres)
            normal_w.append(w_c)
        print(len(normal_w))
        
        normal_w = np.asarray(normal_w)
        new_global_w = np.average(normal_w, axis=0)
        lambda_ = 0.001 # as specified from paper.
        sigma_n = lambda_*s_t
        aggregated_model = client_models[0]
        print(f"new_global_w.shape is: {new_global_w.shape}")
        g_noise = np.random.normal(0, sigma_n, new_global_w.shape[0])
        new_global_w =  (new_global_w + g_noise)
        load_model_weight(aggregated_model, torch.from_numpy(new_global_w.astype(np.float32)).to(device))
        return [aggregated_model],  [1.0]
class FoolsGold(Defense):
    """
    We re-implement FoolsGold defense by extended the original 
    work at https://github.com/DistributedML/FoolsGold
    """
    def __init__(self, num_clients, num_features, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clients = num_clients
        self.n_features = num_features
        self.n_classes = num_classes

    def get_cos_similarity(self, full_deltas):
        '''
        Returns the pairwise cosine similarity of client gradients
        '''
        if True in np.isnan(full_deltas):
            pdb.set_trace()
        return smp.cosine_similarity(full_deltas)

    def importanceFeatureMapGlobal(self, model):
        # aggregate = np.abs(np.sum( np.reshape(model, (10, 784)), axis=0))
        # aggregate = aggregate / np.linalg.norm(aggregate)
        # return np.repeat(aggregate, 10)
        return np.abs(model) / np.sum(np.abs(model))

    def importanceFeatureMapLocal(self, model, topk_prop=0.5):
        # model: np arr
        d = self.n_features # dim of flatten weight
        class_d = int(d / self.n_classes)

        M = model.copy()
        M = np.reshape(M, (self.n_classes, class_d))
        
        # #Take abs?
        # M = np.abs(M)

        for i in range(self.n_classes):
            if (M[i].sum() == 0):
                pdb.set_trace()
            M[i] = np.abs(M[i] - M[i].mean())
            
            M[i] = M[i] / M[i].sum()

            # Top k of 784
            topk = int(class_d * topk_prop)
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            M[i][sig_features_idx] = 0
        
        return M.flatten()   

    def importanceFeatureHard(self, model, topk_prop=0.5):

        class_d = int(self.n_features / self.n_classes)

        M = np.reshape(model, (self.n_classes, class_d))
        importantFeatures = np.ones((self.n_classes, class_d))
        # Top k of 784
        topk = int(class_d * topk_prop)
        for i in range(self.n_classes):
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]     
            importantFeatures[i][sig_features_idx] = 0
        return importantFeatures.flatten()  


    def get_krum_scores(self, X, groupsize):

        krum_scores = np.zeros(len(X))

        # Calculate distances
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(
            X**2, axis=1)[None] - 2 * np.dot(X, X.T)

        for i in range(len(X)):
            krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

        return krum_scores

    def foolsgold(self, this_delta, summed_deltas, sig_features_idx, iter, model, topk_prop=0, importance=False, importanceHard=False, clip=0):
        epsilon = 1e-5
        # Take all the features of sig_features_idx for each clients
        sd = summed_deltas.copy()
        sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

        if importance or importanceHard:
            if importance:
                # smooth version of importance features
                importantFeatures = self.importanceFeatureMapLocal(model, topk_prop)
            if importanceHard:
                # hard version of important features
                importantFeatures = self.importanceFeatureHard(model, topk_prop)
            for i in range(self.n_clients):
                sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)
        N, _ = sig_filtered_deltas.shape
        cs = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    cs[i,i] = 1  
                    continue
                if cs[i,j] != 0 and cs[j,i] != 0:
                    continue
                dot_i = sig_filtered_deltas[i][np.newaxis, :] @ sig_filtered_deltas[j][:, np.newaxis]
                norm_mul = np.linalg.norm(sig_filtered_deltas[i]) * np.linalg.norm(sig_filtered_deltas[j])
                cs[i, j] = cs[j, i] = dot_i / norm_mul
            
        cs = cs - np.eye(N)
        # Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0
        wv = wv / np.max(wv)

        wv[(wv == 1)] = .99

        
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        
        # if iter % 10 == 0 and iter != 0:
        #     print maxcs
        #     print wv

        if clip != 0:

            # Augment onto krum
            scores = self.get_krum_scores(this_delta, self.n_clients - clip)
            bad_idx = np.argpartition(scores, self.n_clients - clip)[(self.n_clients - clip):self.n_clients]

            # Filter out the highest krum scores
            wv[bad_idx] = 0

        avg_updates = np.average(this_delta, axis=0, weights=wv)
        return avg_updates

    def exec(self, client_models, delta, summed_deltas, net_avg, r, device, *args, **kwargs):
        '''
        Aggregates history of gradient directions
        '''
        print(f"START Aggregating history of gradient directions")
        # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(net_avg).detach().cpu().numpy()

        print(f"historical_local_updates.shape is: {summed_deltas.shape}")
        flatten_net_avg = vectorize_net(net_avg).detach().cpu().numpy()


        # Significant features filter, the top k biggest weights
        topk = int(self.n_features / 2)
        sig_features_idx = np.argpartition(flatten_net_avg, -topk)[-topk:]
        sig_features_idx = np.arange(self.n_features)
        avg_delta = self.foolsgold(delta, summed_deltas, sig_features_idx, r, vectorize_avg_net, clip = 0)
        avg_vector_net = vectorize_avg_net + avg_delta
        final_net = client_models[0]
        load_model_weight(final_net, torch.from_numpy(avg_vector_net.astype(np.float32)).to(device))
        return [final_net], [1.0]
class UpperBound(Defense):
    def __init__(self, *args, **kwargs):
        pass
    
    def exec(self, client_models, num_dps, attacker_idxs, g_user_indices, device=torch.device("cuda"), *args, **kwargs):
        #GET KRUM VECTOR
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        
        
        
        # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        selected_idxs = [idx for idx in range(len(client_models)) if idx not in attacker_idxs]
        print("selected_idxs: ", selected_idxs)
        selected_num_dps = np.array(num_dps)[selected_idxs]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]
        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        vectorize_nets = np.asarray(vectorize_nets)[selected_idxs]
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device)) 
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        
        return neo_net_list, neo_net_freq         
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class DeepSight(Defense):
    """
    We re-implement the DeepSight algorithm based on the provided pseudo-code and original paperwork
    by Rieger, Phillip et al. “DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection.” ArXiv abs/2201.00763 (2022): n. pag.
    Link: https://arxiv.org/abs/2201.00763
    """
    def __init__(self, model_name, test_batch_size, *args, **kwargs):
        self.tau = 0.33 # need to verify 
        self.seeds = [1,2,3] # not clear in the paper
        self.total_labels = 10 # may be changed later
        self.input_dim = [28,28] if model_name == 'lenet' else [32,32]
        self.total_samples = 2000
        self.batch_size = test_batch_size
        print(f"self.batch_size: {self.batch_size}")
        self.model = model_name
        
    def calculate_neups(self, g_t, w_i):
        w_diffs = []
        e_updates = []
        g_b, g_w = extract_last_layer(g_t, self.model)
        i_b, i_w = extract_last_layer(w_i, self.model)
        b_diff = i_b-g_b
        total_last_params = len(g_w)
        cnt_params_per_label = int(total_last_params/self.total_labels)
        for k in range(self.total_labels):
            start_i = k*cnt_params_per_label
            end_i = (k+1)*cnt_params_per_label
            w_diff = i_w[start_i, end_i] - g_w[start_i, end_i]
            w_diffs.append(w_diff)
        for k in range(self.total_labels):
            e_update = np.abs(b_diff[k]) + np.sum(np.abs(w_diffs[k]))
            e_updates.append(e_update)
        
        normed_updated = norm(e_updates)**2
        e_updates_normed = [e_updates[i]**2/normed_updated for i in range(self.total_labels)]
        return e_updates_normed
    
    def calculate_TE(self, neups):
        neup_max_indx = np.argmax(neups)
        threshold_eta = max(0.01, 1/self.total_labels)*neups[neup_max_indx]
        threshold_exceeding = np.count_nonzero(neups > threshold_eta)
        return threshold_exceeding
    
    def calculate_ddif(self, g_t, w_i, input_matrix, device):
        # inference step
        output_g_list = []
        output_list = []
        with torch.no_grad():
            for data, target in input_matrix:
                data, target = data.to(device), target.to(device)
                output = w_i(data).to(device).cpu().data.numpy()
                output_g = g_t(data).to(device).cpu().data.numpy()
                # print(f"inferencing, output_g: {output_g}")
                output_g_list.append(output_g)
                output_list.append(output)

        ddif = []
        np_output_list = np.asarray(output_list)
        np_output_list = np_output_list.transpose((-1, 0, 1))
        np_output_list = np_output_list.reshape((np_output_list.shape[0], -1))
        np_output_g_list = np.asarray(output_g_list)
        np_output_g_list = np_output_g_list.transpose((-1, 0, 1))
        np_output_g_list = np_output_g_list.reshape((np_output_g_list.shape[0], -1))
        for i in range(self.total_labels):
            division = np_output_list[i, :] / np_output_g_list[i,:]
            ddif_i = 1/self.total_samples* np.sum(division)
            ddif.append(ddif_i)
        # print(f"ddif: {ddif}")
        return ddif

    def distsFromClust(self, clusters, N):
        pairwise_dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                pairwise_dists[i,j] = 0 if clusters[i] == clusters[j] else 1
        return pairwise_dists
    
    def clustering(self, N, neups, ddifs, cosine_distances):
        cs_distance_matrix = cosine_distances
        hdbscan_cs = hdbscan.HDBSCAN(metric='precomputed')
        hdbscan_cs.fit(cs_distance_matrix)
        cosine_clusters = hdbscan_cs.labels_
        cosine_clusters_dists = self.distsFromClust(cosine_clusters, N)
        hdbscan_neups = hdbscan.HDBSCAN(algorithm='best')
        hdbscan_neups.fit(neups)
        neup_clusters = hdbscan_neups.labels_
        neup_cluster_dists = self.distsFromClust(neup_clusters, N)
        ddif_clusters_dists_list = []
        for ddif in ddifs:
            if len(ddif) > 0:
                ddif_cluster = hdbscan.HDBSCAN(algorithm='best')
                ddif_cluster.fit(ddif)
                ddif_clusters = ddif_cluster.labels_
                ddif_clusters_dists = self.distsFromClust(ddif_clusters, N)
                ddif_clusters_dists_list.append(ddif_clusters_dists)
        if len(ddif_clusters_dists) > 0:
            merged_ddif_clust_dists = np.average(ddif_clusters_dists_list, axis=0)
            merged_distances = np.mean(np.array([ merged_ddif_clust_dists, neup_cluster_dists,  cosine_clusters_dists]), axis=0 )
        else:
            merged_distances = np.mean(np.array([neup_cluster_dists,  cosine_clusters_dists]), axis=0)

        clusters = hdbscan.HDBSCAN(metric='precomputed').fit(merged_distances)        
        final_clusters = clusters.labels_
        return final_clusters

    def exec(self, client_models, g_t, num_dps, input_dim, g_user_indices, selected_attackers, model_name, device, *args, **kwargs):
        w = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models] # list of N received local models
        n = len(w) # number of models
        # input_dim: dimension of a single input
        cosine_distances = np.zeros((n,n))
        g_bias, g_weight = extract_last_layer(g_t, self.model)
        for i in range(n):
            for j in range(n):
                i_b, i_w = extract_last_layer(client_models[i], self.model)
                j_b, j_w = extract_last_layer(client_models[j], self.model)
                update_i = i_b - g_bias
                update_j = j_b - g_bias
                cosine_distances[i,j]= 1.0 - dot(update_i, update_j)/(norm(update_i)*norm(update_j))
        neups_list = [self.calculate_neups(g_t, client_models[i]) for i in range(n)]
        te_list = [self.calculate_TE(neups_list[i]) for i in range(n)]
        input_matrices = []
        for s in self.seeds:
            np.random.seed(s)
            if self.model == "lenet":
                noise_dataset = datasets.FakeData(size=self.total_samples, image_size=(28, 28), transform=transforms.Compose([transforms.ToTensor()]), random_offset=s)
            else:
                noise_dataset = datasets.FakeData(size=self.total_samples, image_size=(3, 32, 32), transform=transforms.Compose([transforms.ToTensor()]), random_offset=s)
                
            
            input_matrices.append(noise_dataset)
        ddifs_list = []
        for input_matrix in input_matrices:
            dataloader = torch.utils.data.DataLoader(input_matrix, batch_size=self.batch_size)
            ddifs = [self.calculate_ddif(g_t, client_model, dataloader, device) for client_model in client_models]
            ddifs_list.append(ddifs)

        # first classification layer:
        labels = [1 if te_list[i] <= np.median(te_list)/2 else 0 for i in range(n)]
        # print(f"te_list: {te_list} {np.median(te_list)/2}")
        # print(f"labels: {labels}")
        final_clusters = self.clustering(n, neups_list, ddifs_list, cosine_distances)
        final_clusters = np.asarray(final_clusters)
        # print(f"final_clusters: {final_clusters}")
        cluster_list = np.unique(final_clusters)
        acpt_models_idxs = []
        labels = np.asarray(labels)
        for cluster in cluster_list:
            if cluster == -1:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                for i in indexes:
                    if labels[i] == 1:
                        continue
                    else:
                        acpt_models_idxs.append(i)
            else:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                amount_of_positives = np.sum(labels[indexes])/len(indexes)
                if amount_of_positives < self.tau:
                    for idx in indexes:
                        acpt_models_idxs.append(idx)
        
        if len(acpt_models_idxs) > 0:
            print(f"acpt_models_idxs: {acpt_models_idxs}")
            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[acpt_models_idxs]
            reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(acpt_models_idxs, [g_user_indices[ti] for ti in acpt_models_idxs]))
            #aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0)

            # clipping layer
            flatten_g_t = vectorize_net(g_t).detach().cpu().numpy()
            local_models_norms = [norm(w[i]-flatten_g_t) for i in range(n)]
            s = np.median(local_models_norms)
            lambda_idxs = []
            for idx in range(n):
                vectorize_diff = w[idx] - flatten_g_t
                weight_diff_norm = norm(vectorize_diff)
                lambda_idx = min(1.0, s/weight_diff_norm)
                w[idx] = lambda_idx*w[idx]
            if not acpt_models_idxs:
                return [g_t], [1.0]
            aggregated_grad = np.average(np.array(w)[acpt_models_idxs, :], weights=reconstructed_freq, axis=0).astype(np.float32)

            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
            neo_net_list = [aggregated_model]
            #logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
        else:
            neo_net_freq = [1.0]
            neo_net_list = [g_t]
        return neo_net_list, neo_net_freq
    
if __name__ == "__main__":
    # some tests here
    import copy
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check 1, this should recover the global model
    sim_global_model = Net().to(device)
    sim_local_model1 = copy.deepcopy(sim_global_model)
    #sim_local_model = Net().to(device)
    defender = WeightDiffClippingDefense(norm_bound=5)
    defender.exec(client_model=sim_local_model1, global_model=sim_global_model)

    vec_global_sim_net = vectorize_net(sim_global_model)
    vec_local_sim_net1 = vectorize_net(sim_local_model1)

    # Norm Weight Diff: 0.0, Norm Clipped Weight Diff 0.0
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 8.843663215637207    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(), 
        torch.norm(vec_local_sim_net1).item()))

    # check 2, adding some large perturbation
    sim_local_model2 = copy.deepcopy(sim_global_model)
    scaling_facor = 2
    for p_index, p in enumerate(sim_local_model2.parameters()):
        p.data = p.data + torch.randn(p.size()) * scaling_facor
    defender.exec(client_model=sim_local_model2, global_model=sim_global_model)
    vec_local_sim_net2 = vectorize_net(sim_local_model2)

    # Norm Weight Diff: 2191.04345703125, Norm Clipped Weight Diff 4.999983787536621
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 10.155366897583008    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(), 
        torch.norm(vec_local_sim_net2).item()))
