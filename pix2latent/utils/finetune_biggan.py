import copy
import torch
import utils.loss_functions as LF



scale_lower = {'embeddings'  : 0.0,
               'gen_z'       : 0.0,
               'layers.0'    : 0.1,
               'layers.1'    : 0.1,
               'layers.2'    : 0.2,
               'layers.3'    : 0.3,
               'layers.4'    : 0.4,
               'layers.5'    : 0.5,
               'layers.6'    : 0.6,
               'layers.7'    : 0.7,
               'layers.8'    : 0.8,
               'layers.9'    : 0.9,
               'layers.10'   : 1.0,
               'layers.11'   : 1.0,
               'layers.12'   : 1.0,
               'layers.13'   : 1.0,
               'conv_to_rgb' : 1.0,
               }


scale_upper = {'embeddings'  : 1.0,
               'gen_z'       : 1.0,
               'layers.0'    : 1.0,
               'layers.1'    : 1.0,
               'layers.2'    : 1.0,
               'layers.3'    : 1.0,
               'layers.4'    : 1.0,
               'layers.5'    : 1.0,
               'layers.6'    : 0.9,
               'layers.7'    : 0.8,
               'layers.8'    : 0.8,
               'layers.9'    : 0.6,
               'layers.10'   : 0.5,
               'layers.11'   : 0.4,
               'layers.12'   : 0.3,
               'layers.13'   : 0.2,
               'conv_to_rgb' : 0.1,
               }



def set_finetune(model):
    # train all weights except batch norm parameters
    model.eval()
    for k, v in model.named_parameters():
        if 'bn' in k:
            v.requires_grad = False
        else:
            v.requires_grad = True
    return


def map_param_to_weight(model_state_dict, weight_scale):
    if weight_scale is not None:
        param_to_weight = {}
        for param_name, curr_param in model_state_dict.named_parameters():
            if 'bn' in param_name:
                continue

            found = False
            for k, w in weight_scale.items():
                if k in param_name:
                    param_to_weight[param_name] = w
                    found = True

            if not found:
                print('no weight found for scaling: {}'.format(param_name))
    else:
        w = 1.0
    return param_to_weight


def finetune(model, z, cv, target, mask, iter=500, lr=1e-5, step_lr=None,
             rec_weight=1, reg_weight=5e2, per_weight=10.0,
             weight_scale=scale_upper, use_perceptual_loss=True,
             automatic_schedule=False, epsilon=0.1, track=False):

    rloss_fn = LF.ReconstructionLoss()

    if use_perceptual_loss:
        ploss_fn = LF.PerceptualLoss(net='vgg')

    # train all weights except batch norm parameters
    orig_model = copy.deepcopy(model)

    set_finetune(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    param_to_weight = None
    if weight_scale is not None:
        param_to_weight = map_param_to_weight(orig_model, weight_scale)

    results = []

    i = 0
    while True:
        if not automatic_schedule:
            if i >= iter:
                break
        else:
            if step_lr is not None:
                if i != 0 and i % step_lr == 0:
                    reg_weight = reg_weight / 10.

        opt.zero_grad()
        out = model(z=z, c=cv, embed_class=False, truncation=1.0)

        if track:
            results.append(out.clone().detach().cpu())

        rec_loss = rloss_fn(out, target, mask)

        reg_loss = LF.weight_regularization(orig_model=orig_model,
                                            curr_model=model,
                                            weight_dict=param_to_weight)

        if use_perceptual_loss:
            per_loss = ploss_fn(out, target, mask).mean()
        else:
            per_loss = 0.0

        loss = (rec_weight * rec_loss) +\
               (reg_weight * reg_loss) +\
               (per_weight * per_loss)

        loss.backward()
        opt.step()

        if automatic_schedule:
            if rec_loss.item() < epsilon:
                break

        if i % 100 == 0:
            print('[{}] rec: {} reg: {}'.format(
                                    i, rec_loss.item(), reg_loss.item()))
        i += 1
    return out, model, results
