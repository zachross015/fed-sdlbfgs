import copy
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_from_z(model, z, t, reduction):
    if model.name_loss == 'multi-class classification':
        loss = F.cross_entropy(z, t, reduction = reduction)
    elif model.name_loss == 'binary classification':
        loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float().unsqueeze_(1))
        if reduction == 'none':
            loss = loss.squeeze(1)
        
    elif model.name_loss == 'logistic-regression':
        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            loss = loss / z.size(0) / z.size(1)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            
    elif model.name_loss == 'logistic-regression-sum-loss':
        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            loss = loss / z.size(0)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())

    elif model.name_loss == 'linear-regression-half-MSE':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t) / 2
            loss = loss / z.size(0)
        
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t) / 2
            loss = torch.sum(loss, dim=1)
            
    elif model.name_loss == 'linear-regression':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t)
            loss = loss / z.size(0)
            
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t)
            loss = torch.sum(loss, dim=1)
    
    return loss

def get_zero_torch(params):
    layers_params = params['layers_params']
    device = params['device']
    
    delta = []
    for l in range(len(layers_params)):
        delta_l = {}
        delta_l['W'] = torch.zeros(layers_params[l]['output_size'], layers_params[l]['input_size'], device=device)
        delta_l['b'] = torch.zeros(layers_params[l]['output_size'], device=device)
        delta.append(delta_l)
        
    return delta

def get_subtract(model_grad, delta, params):
    diff_p = get_zero(params)
    for l in range(params['numlayers']):
        for key in diff_p[l]:
            diff_p[l][key] = np.subtract(model_grad[l][key], delta[l][key])
    return diff_p

def get_subtract_torch(model_grad, delta):
    diff_p = []
    for l in range(len(model_grad)):
        diff_p_l = {}
        for key in model_grad[l]:
            diff_p_l[key] = torch.sub(model_grad[l][key], delta[l][key])
        diff_p.append(diff_p_l)
    return diff_p



def get_plus(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = np.add(model_grad[l][key], delta[l][key])
        sum_p.append(sum_p_l)
    return sum_p

def get_plus_torch(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + delta[l][key]
        sum_p.append(sum_p_l)
    return sum_p

def get_if_nan(p):
    for l in range(len(p)):
        for key in p[l]:
            if torch.sum(p[l][key] != p[l][key]):
                return True
    return False



def get_torch_tensor(p, params):
    p_torch = []
    for l in range(len(p)):
        p_torch_l = {}
        for key in p[l]:
            p_torch_l[key] = torch.from_numpy(p[l][key]).to(params['device'])
        p_torch.append(p_torch_l)
    return p_torch

def get_plus_scalar(alpha, model_grad):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + alpha
        sum_p.append(sum_p_l)
    return sum_p

def get_multiply_scalar(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_no_grad(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key].data
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_blockwise(alpha, delta, params):
    alpha_p = []
    for l in range(params['numlayers']):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha[l] * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_torch(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = torch.mul(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = np.multiply(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_weighted_sum_batch(hat_v, batch_grads_test, params):
    alpha_p = get_zero(params)
    for l in range(params['numlayers']):
        alpha_p['W'][l] = np.sum(hat_v[:, None, None] * batch_grads_test['W'][l], axis=0)
        alpha_p['b'][l] = np.sum(hat_v[:, None] * batch_grads_test['b'][l], axis=0)
    return alpha_p

def get_opposite(delta):
    numlayers = len(delta)
    p = []
    for l in range(numlayers):
        p_l = {}
        for key in delta[l]:
            p_l[key] = -delta[l][key]
        p.append(p_l)
        
    return p

def get_model_grad(model, params):
    model_grad_torch = []
    for l in range(model.numlayers):
        model_grad_torch_l = {}
        for key in model.layers_weight[l]:
            model_grad_torch_l[key] = copy.deepcopy(model.layers_weight[l][key].grad)
        model_grad_torch.append(model_grad_torch_l)
    return model_grad_torch


def get_regularized_loss_and_acc_from_x_whole_dataset(model, x, t, reduction, params):
    N1 = params['N1']
    N1 = np.minimum(N1, len(x))
    
    i = 0
    device = params['device']
    
    list_loss = []
    list_acc = []
    
    model.eval()
    
    while i + N1 <= len(x):
#         with torch.no_grad():
        z, test_a, test_h = model.forward(torch.from_numpy(x[i: i+N1]).to(device))
            
        torch_t_mb = torch.from_numpy(t[i: i+N1]).to(params['device'])
        list_loss.append(
            get_regularized_loss_from_z(model, z, torch_t_mb,
                reduction, params['tau']).item())
        list_acc.append(
            get_acc_from_z(model, params, z, torch_t_mb))
       
        i += N1
    model.train()
    
    return sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)

def get_regularized_loss_from_z(model, z, t, reduction, tau):
    loss = get_loss_from_z(model, z, t, reduction)
    loss += 0.5 * tau *\
    get_dot_product_torch(model.layers_weight, model.layers_weight)
    return loss

def get_if_stop(args, i, iter_per_epoch, timesCPU):
    if args['if_max_epoch']:
        if i < int(args['max_epoch/time'] * iter_per_epoch):
            return False
        else:
            return True
    else:
        if timesCPU[-1] < args['max_epoch/time']:
            return False
        else:
            return True
        
        
def get_square(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.square(delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_square_torch(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.mul(delta_1[l][key], delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_sqrt(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_sqrt_torch(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_max_with_0(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = F.relu(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_divide(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.divide(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def get_divide_torch(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.div(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def get_dot_product_torch(delta_1, delta_2):
    dot_product = 0
    for l in range(len(delta_1)):
        for key in delta_1[l]:
            dot_product += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
    return dot_product

def get_dot_product_blockwise_torch(delta_1, delta_2):
    dot_product = []
    for l in range(len(delta_1)):
        dot_product_l = 0
        for key in delta_1[l]:
            dot_product_l += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
        dot_product.append(dot_product_l)
    return dot_product

def get_dot_product_batch(model_grad, batch_grads_test, params):
    # numlayers = params['numlayers']
    
    dot_product = np.zeros(len(batch_grads_test['W'][0]))
    for l in range(params['numlayers']):
        dot_product += np.sum(
            np.sum(np.multiply(model_grad['W'][l][None, :], batch_grads_test['W'][l]), axis=-1), axis=-1)
        dot_product += np.sum(np.multiply(model_grad['b'][l][None, :], batch_grads_test['b'][l]), axis=-1)
    
    return dot_product

def get_acc_from_z(model, params, z, torch_t):
    if model.name_loss == 'multi-class classification':
        y = z.argmax(dim=1)
        acc = torch.mean((y == torch_t).float())
        
    elif model.name_loss == 'binary classification':
        z_1 = torch.sigmoid(z)
        y = (z_1 > 0.5)
        y = y[:, 0]
        acc = np.mean(y.cpu().data.numpy() == np_t)
    elif model.name_loss in ['logistic-regression',
                             'logistic-regression-sum-loss']:
        z_sigmoid = torch.sigmoid(z)
        criterion = nn.MSELoss(reduction = 'mean')
        acc = criterion(z_sigmoid, torch_t)
  
    elif model.name_loss in ['linear-regression',
                             'linear-regression-half-MSE']:
        acc = nn.MSELoss(reduction = 'mean')(z, torch_t)

    
    acc = acc.item()
    
    return acc

def get_homo_grad(model_grad_N1, params):
    device = params['device']

    homo_model_grad_N1 = []
    for l in range(params['numlayers']):
        homo_model_grad_N1_l = torch.cat((model_grad_N1[l]['W'], model_grad_N1[l]['b'].unsqueeze(1)), dim=1)
        homo_model_grad_N1.append(homo_model_grad_N1_l)

    return homo_model_grad_N1  
    



def Kron_BFGS_update(data_, params):
    i = params['i']
    numlayers = params['numlayers']
    data_['model_homo_grad_used_torch'] = get_homo_grad(data_['model_grad_used_torch'], params)

    delta = []
    for l in range(numlayers):
        action_h = params['Kron_BFGS_action_h']
        action_a = params['Kron_BFGS_action_a']
        step_ = 1
        delta_l, data_ = Kron_BFGS_update_per_layer(data_, params, l, action_h, action_a, step_)
        delta.append(delta_l)

    p = get_opposite(delta)
    data_['p_torch'] = p

    # get next grad
    model_new = copy.deepcopy(data_['model'])
    for l in range(model_new.numlayers):
        for key in model_new.layers_weight[l]:
            model_new.layers_weight[l][key].data += params['alpha'] * p[l][key].data

    z_next, a_next, h_next = model_new.forward(data_['X_mb'])
    loss = get_loss_from_z(model_new, z_next, data_['t_mb'], reduction='mean')

    model_new.zero_grad()
    loss.backward()

    a_grad_next = [len(data_['X_mb']) * (a_l.grad) for a_l in a_next]
    data_['a_grad_next'] = a_grad_next
    data_['h_next'] = h_next
    data_['a_next'] = a_next

    for l in range(numlayers):
        action_h = params['Kron_BFGS_action_h']
        action_a = params['Kron_BFGS_action_a']
        step_ = 2
        _, data_ = Kron_BFGS_update_per_layer(data_, params, l, action_h, action_a, step_)


    return data_, params

def Kron_BFGS_update_per_layer(data_, params, l, action_h, action_a, step_):
    i = params['i']
    algorithm = params['algorithm']
    N1 = params['N1']

    if step_ == 1:
        device = params['device']
        model_homo_grad = data_['model_homo_grad_used_torch']
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]

        if i == 0:
            Kron_BFGS_matrices_l['H'] = {}
            if action_h in ['Hessian-action-BFGS']:
                Kron_BFGS_matrices_l['H']['h'] = torch.eye(data_['h_N2'][l].size()[1], device=device)

            if action_a == 'BFGS':
                Kron_BFGS_matrices_l['H']['a_grad'] = torch.eye(
                data_['a_grad_N2'][l].size()[1], device=device, requires_grad=False)

                Kron_BFGS_matrices_l['H']['a_grad'] *= params['Kron_LBFGS_Hg_initial']

        if action_h in ['Hessian-action-BFGS', 'Hessian-action-LBFGS']:
            # update A
            A_l = Kron_BFGS_matrices_l['A']
            if params['N1'] < params['num_train_data'] and i == 0:
                1
            else:
                beta_ = params['Kron_BFGS_A_decay']
                homo_h_l = torch.cat((data_['h_N2'][l], torch.ones(N1, 1, device=device)), dim=1)

                decay_ = beta_
                weight_ = 1-beta_

                A_l = decay_ * A_l + weight_ * torch.mm(homo_h_l.t(), homo_h_l).data / data_['h_N2'][l].size()[0]

            Kron_BFGS_matrices_l['A'] = A_l

            if action_h in ['Hessian-action-BFGS', 'Hessian-action-LBFGS']:
                epsilon_ = params['Kron_BFGS_A_LM_epsilon']

                A_l_LM = Kron_BFGS_matrices_l['A'] + epsilon_ * torch.eye(Kron_BFGS_matrices_l['A'].size(0), device=device)
                Kron_BFGS_matrices_l['A_LM'] = A_l_LM

                if action_h in ['Hessian-action-BFGS']:
                    if i == 0:
                        Kron_BFGS_matrices_l['H']['h'] = A_l_LM.inverse()

        data_['Kron_BFGS_matrices'][l] = Kron_BFGS_matrices_l
        delta_l = Kron_BFGS_compute_direction(model_homo_grad, l, data_, params)
        return delta_l, data_

    elif step_ == 2:

        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]

        a_grad_next = data_['a_grad_next']
        a_next = data_['a_next']

        if action_a == 'BFGS':
            H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']

        # compute s
        s_l_a = torch.mean(data_['a_N2'][l], dim=0).data - torch.mean(a_next[l], dim=0).data

        # compute y
        mean_a_grad_l = torch.mean(data_['a_grad_N2'][l], dim=0)
        mean_a_grad_next_l = torch.mean(a_grad_next[l], dim=0)
        y_l_a = mean_a_grad_l - mean_a_grad_next_l


        if N1 < params['num_train_data']:
            decay_ = 0.9
        else:
            decay_ = 0

        data_['Kron_BFGS_momentum_s_y'][l]['s'] =\
        decay_ * data_['Kron_BFGS_momentum_s_y'][l]['s'] + (1-decay_) * s_l_a

        data_['Kron_BFGS_momentum_s_y'][l]['y'] =\
        decay_ * data_['Kron_BFGS_momentum_s_y'][l]['y'] + (1-decay_) * y_l_a

        s_l_a = data_['Kron_BFGS_momentum_s_y'][l]['s']
        y_l_a = data_['Kron_BFGS_momentum_s_y'][l]['y']

        s_l_a, y_l_a = kron_bfgs_update_damping(s_l_a, y_l_a, l, data_, params)

        if params['Kron_BFGS_action_a'] == 'LBFGS':

            data_['Kron_LBFGS_s_y_pairs']['a'][l] =\
            Kron_LBFGS_append_s_y(
                s_l_a,
                y_l_a,
                data_['Kron_LBFGS_s_y_pairs']['a'][l],
                mean_a_grad_l,
                params['Kron_LBFGS_Hg_initial'],
                params
            )

        elif params['Kron_BFGS_action_a'] == 'BFGS':
            Kron_BFGS_matrices_l['H']['a_grad'], update_status =\
        get_BFGS_formula(H_l_a_grad, s_l_a, y_l_a, mean_a_grad_l)


        if action_h in ['Hessian-action-BFGS', 'Hessian-action-LBFGS']:
            mean_h_l = torch.mean(data_['h_N2'][l], dim=0).data

#             mean_h_l = torch.cat((mean_h_l, torch.mean(mean_h_l).unsqueeze(0)), dim=0)
            mean_h_l = torch.cat(
                                (mean_h_l, torch.ones(1, device=params['device'])),
                                dim=0
                            )

            if action_h == 'Hessian-action-LBFGS':
                s_l_h = LBFGS_Hv(
                    mean_h_l,
                    data_['Kron_LBFGS_s_y_pairs']['h'][l],
                    params
                )
            elif action_h in ['Hessian-action-BFGS']:

                H_l_h = Kron_BFGS_matrices_l['H']['h']
                s_l_h = torch.mv(H_l_h, mean_h_l)


            y_l_h = torch.mv(Kron_BFGS_matrices_l['A_LM'], s_l_h)
            if action_h == 'Hessian-action-LBFGS':
                data_['Kron_LBFGS_s_y_pairs']['h'][l] =\
                Kron_LBFGS_append_s_y(
                    s_l_h,
                    y_l_h,
                    data_['Kron_LBFGS_s_y_pairs']['h'][l],
                    [],
                    params['Kron_LBFGS_Ha_initial'],
                    params
                )
            elif action_h in ['Hessian-action-BFGS']:
                Kron_BFGS_matrices_l['H']['h'], update_status =\
                get_BFGS_formula(H_l_h, s_l_h, y_l_h, mean_h_l)

        elif action_h == 'BFGS':
            h_next = data_['h_next']

            H_l_h = Kron_BFGS_matrices_l['H']['h']


            # compute s
            mean_h_l = torch.mean(data_['h_N2'][l], dim=0).data

            s_l_h = torch.mv(H_l_h, mean_h_l)
            s_l_h = s_l_h * np.sqrt(params['alpha'])

            # compute y
            mean_h_next_l = torch.mean(h_next[l], dim=0).data
            y_l_h = mean_h_l - mean_h_next_l


            Kron_BFGS_matrices_l['H']['h'] = get_BFGS_formula(H_l_h,
                                                   s_l_h, y_l_h,
                                                      mean_h_l)

        data_['Kron_BFGS_matrices'][l] = Kron_BFGS_matrices_l

        return [], data_

def get_BFGS_PowellHDamping(s_l_a, y_l_a, alpha, l, data_, params):
    Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]

    if params['Kron_BFGS_action_a'] == 'LBFGS':
        1
    elif params['Kron_BFGS_action_a'] == 'BFGS':
        H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']

    s_T_y = torch.dot(s_l_a, y_l_a)
    if params['Kron_BFGS_action_a'] == 'LBFGS':
        Hy = LBFGS_Hv(
            y_l_a,
            data_['Kron_LBFGS_s_y_pairs']['a'][l],
            params
        )
    elif params['Kron_BFGS_action_a'] == 'BFGS':
        Hy = torch.mv(H_l_a_grad ,y_l_a)


    yHy = torch.dot(y_l_a, Hy)
    sy_over_yHy_before = s_T_y.item() / yHy.item()

    if sy_over_yHy_before > alpha:
        theta = 1
        damping_status = 0
    else:
        theta =  ((1-alpha) * yHy / (yHy - s_T_y)).item()
        original_s_l_a = s_l_a
        s_l_a = theta * s_l_a + (1-theta) * Hy
        damping_status = 1

    return s_l_a, y_l_a, sy_over_yHy_before

def kron_bfgs_update_damping(s_l_a, y_l_a, l, data_, params):
    s_l_a, y_l_a, _ = get_BFGS_PowellHDamping(s_l_a, y_l_a, 0.2, l, data_, params)
    s_l_a, y_l_a = get_BFGS_ModifiedDamping(s_l_a, y_l_a, l, data_, params)

    return s_l_a, y_l_a


def get_BFGS_ModifiedDamping(s_l_a, y_l_a, l, data_, params):
    alpha = params['Kron_BFGS_H_epsilon']

    s_T_s = torch.dot(s_l_a, s_l_a)
    s_T_y = torch.dot(s_l_a, y_l_a)

    if s_T_y / s_T_s > alpha:
        damping_status = 0
    else:
        theta =  (1-alpha) * s_T_s / (s_T_s - s_T_y)
        y_l_a = theta * y_l_a + (1-theta) * s_l_a
        damping_status = 1

    return s_l_a, y_l_a


def Kron_BFGS_compute_direction(model_homo_grad, l, data_, params):
    delta_l = {}

    if params['Kron_BFGS_action_a'] == 'LBFGS':
        delta_l_W = LBFGS_Hv(
                model_homo_grad[l],
                data_['Kron_LBFGS_s_y_pairs']['a'][l],
                params
            )

    elif params['Kron_BFGS_action_a'] == 'BFGS':
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']
        delta_l_W = torch.mm(H_l_a_grad, model_homo_grad[l])


    if params['Kron_BFGS_action_h'] in ['LBFGS','Hessian-action-LBFGS']:
        delta_l_W = LBFGS_Hv(
                delta_l_W.t(),
                data_['Kron_LBFGS_s_y_pairs']['h'][l],
                params
            )
        delta_l_W = delta_l_W.t()
    elif params['Kron_BFGS_action_h'] == 'inv':
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        H_l_h = Kron_BFGS_matrices_l['A_inv']
        delta_l_W = torch.mm(delta_l_W, H_l_h)
    elif params['Kron_BFGS_action_h'] in ['Hessian-action-BFGS']:

        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        H_l_h = Kron_BFGS_matrices_l['H']['h']
        delta_l_W = torch.mm(delta_l_W, H_l_h)

    delta_l['W'] = delta_l_W[:, :-1]
    delta_l['b'] = delta_l_W[:, -1]

    return delta_l

def LBFGS_Hv(v, s_y_pairs, params):
    list_s = s_y_pairs['s']
    list_y = s_y_pairs['y']
    R_inv = s_y_pairs['R_inv']
    yTy = s_y_pairs['yTy']
    D_diag = s_y_pairs['D_diag']

    gamma = s_y_pairs['gamma']

    left_matrix = s_y_pairs['left_matrix']
    right_matrix = s_y_pairs['right_matrix']

    if len(list_s) == 0:
        Hv = v
    else:
        device = params['device']
        len_v_size = len(v.size())
        if len_v_size == 1:
            v = v.unsqueeze(1)

        if gamma == -1:
            gamma = 1 / R_inv[-1][-1].item() / yTy[-1][-1].item()
        assert gamma > 0

        Hv = gamma * v + torch.mm(left_matrix, torch.mm(right_matrix, v))
        if len_v_size == 1:
            Hv = Hv.squeeze(1)
    return Hv

def Kron_LBFGS_append_s_y(s, y, s_y_pairs, g_k, gamma, params):
    s = s.unsqueeze(1)
    y = y.unsqueeze(1)

    device = params['device']

    if len(g_k) == 0:
        dot_gk_gk = 0
    else:
        dot_gk_gk = torch.mm(g_k.unsqueeze(0), g_k.unsqueeze(1)).item()

    dot_new_y_new_s = torch.mm(y.t(), s)
    dot_new_s_new_s = torch.mm(s.t(), s)

    if (not np.isinf(dot_new_s_new_s.item())) and\
    dot_new_y_new_s.item() > 10**(-4) * dot_new_s_new_s.item() * np.sqrt(dot_gk_gk):

        if len(s_y_pairs['s']) == params['Kron_BFGS_number_s_y']:
            s_y_pairs['R_inv'] = s_y_pairs['R_inv'][1:, 1:]

            s_y_pairs['yTy'] = s_y_pairs['yTy'][1:, 1:]

            s_y_pairs['s'] = s_y_pairs['s'][1:]
            s_y_pairs['y'] = s_y_pairs['y'][1:]

            s_y_pairs['D_diag'] = s_y_pairs['D_diag'][1:]

        if len(s_y_pairs['s']) == 0:
            s_y_pairs['s'] = s.t()
            s_y_pairs['y'] = y.t()
        else:

            s_y_pairs['s'] = torch.cat((s_y_pairs['s'], s.t()), dim=0)
            s_y_pairs['y'] = torch.cat((s_y_pairs['y'], y.t()), dim=0)

        if len(s_y_pairs['yTy']) == 0:
            s_y_pairs['yTy'] = torch.mm(s_y_pairs['y'], s_y_pairs['y'].t())
        else:
            yT_new_y = torch.mm(s_y_pairs['y'], y)

            s_y_pairs['yTy'] = torch.cat((s_y_pairs['yTy'], yT_new_y[:-1]), dim=1)
            s_y_pairs['yTy'] = torch.cat((s_y_pairs['yTy'], yT_new_y.t()), dim=0)

        if len(s_y_pairs['s']) == 1:
            s_y_pairs['D_diag'] = torch.mm(s_y_pairs['s'], s_y_pairs['y'].t())
            s_y_pairs['D_diag'] = s_y_pairs['D_diag'].squeeze(0)

            s_y_pairs['R_inv'] = 1 / s_y_pairs['D_diag'][-1]
            s_y_pairs['R_inv'] = s_y_pairs['R_inv'].unsqueeze(0).unsqueeze(1)

        else:
            sT_y = torch.mm(s_y_pairs['s'], y)

            s_y_pairs['D_diag'] = torch.cat(
                (s_y_pairs['D_diag'], sT_y[-1]), dim=0
            )

            B_22 = 1 / sT_y[-1][-1].item()
            B_22 = torch.tensor(B_22, device=device)


            B_22 = B_22.unsqueeze(0)
            B_22 = B_22.unsqueeze(1)

            s_y_pairs['R_inv'] = torch.cat(
                (
                    torch.cat(
                        (s_y_pairs['R_inv'], torch.zeros(1, s_y_pairs['R_inv'].size(1), device=params['device'])),
                        dim=0
                    ),
                    torch.cat((-B_22 * torch.mm(s_y_pairs['R_inv'], sT_y[:-1]), B_22), dim=0)
                ),
                dim=1
            )

        if gamma == -1:
            gamma = s_y_pairs['D_diag'][-1].item() / s_y_pairs['yTy'][-1][-1].item()


        s_y_pairs['gamma'] = gamma
        R_inv_sT = torch.mm(s_y_pairs['R_inv'], s_y_pairs['s'])

        if len(s_y_pairs['right_matrix']) < 2 * params['Kron_BFGS_number_s_y']:
            s_y_pairs['left_matrix'] = torch.cat(
                (R_inv_sT.t(), gamma * s_y_pairs['y'].t()), dim=1
            )
            s_y_pairs['right_matrix'] = torch.cat(
                (
                    torch.mm(torch.diag(s_y_pairs['D_diag']) + gamma * s_y_pairs['yTy'], R_inv_sT) - gamma * s_y_pairs['y'],
                    - R_inv_sT
                ), dim=0
            )

        else:
            m = params['Kron_BFGS_number_s_y']
            s_y_pairs['left_matrix'][:, :m] = R_inv_sT.t()
            s_y_pairs['left_matrix'][:, m:] = gamma * s_y_pairs['y'].t()
            s_y_pairs['right_matrix'][:m] = s_y_pairs['D_diag'][:, None] * R_inv_sT + gamma * (torch.mm(s_y_pairs['yTy'],R_inv_sT) - s_y_pairs['y'])
            s_y_pairs['right_matrix'][m:] = - R_inv_sT

    return s_y_pairs

def get_BFGS_formula(H, s, y, g_k):
    s = s.data
    y = y.data
    rho_inv = torch.dot(s, y)

    if rho_inv <= 0:
        return H, 1
    elif rho_inv <= 10**(-4) * torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item()):
        return H, 2

    rho = 1 / rho_inv

    Hy = torch.mv(H, y)
    H_new = H.data +\
    (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
    rho * (torch.ger(s, Hy) + torch.ger(Hy, s))

    if torch.max(torch.isinf(H_new)):
        return H, 4
    else:
        H = H_new

    return H, 0

