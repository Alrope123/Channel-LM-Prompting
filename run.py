import os
import torch
import numpy as np

from tqdm import tqdm

from model_util import get_optimizer_and_scheduler, get_dataloader

def train(logger, model, inputs, batch_size, output_dir, local_rank,
          prior_inputs=None,
          learning_rate=1e-5,
          regularization_weight=0.0,
          aux_weight=0.0,
          target_indices=None,
          warmup_steps=50,
          num_training_steps=200,
          gradient_accumulation_steps=1,
          max_grad_norm=1.0,
          eval_period=20,
          prompt_tune=False,
          head_tune=False,
          transform_tune=False,
          prior_tune=False,
          bad=False):
    optimizer, scheduler = get_optimizer_and_scheduler(
        "adamw",
        model.named_parameters(),
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()
    dataloader = get_dataloader(inputs, batch_size, is_training=True, prior_inputs=prior_inputs)

    n_trainable_params = len([param for param in model.parameters() if param.requires_grad])
    n_gpus = torch.cuda.device_count()
    logger.info("Training {} parameters on {} examples for {} steps using {} GPUs".format(
        n_trainable_params, len(inputs["input_ids"]), num_training_steps, n_gpus))

    model.train()

    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Start training")
    for epoch in range(num_training_steps):
        for batch in dataloader:
            global_step += 1

            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            classes=batch[3].cuda()
            prior_input_ids=None
            prior_attention_mask=None
            prior_token_type_ids=None

            if len(batch)==4:
                labels=None
            elif len(batch)==7:
                labels=None
                prior_input_ids=batch[4].cuda()
                prior_attention_mask=batch[5].cuda()
                prior_token_type_ids=batch[6].cuda()
            elif len(batch)==5:
                labels=batch[4].cuda()
            else:
                labels=batch[7].cuda()

            with torch.cuda.amp.autocast():
                loss = run_model(model, input_ids, attention_mask, token_type_ids, regularization_weight, classes=classes,
                                 aux_weight=aux_weight, target_indices=target_indices,
                                 prior_input_ids=prior_input_ids, prior_attention_mask=prior_attention_mask, prior_token_type_ids=prior_token_type_ids,
                                 labels=labels, bad=bad, local_rank=local_rank)
                loss = loss.mean()

            if torch.isnan(loss).data:
                print ("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            scaler.scale(loss).backward()
            if global_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)    # We have accumulated enought gradients
                model.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if global_step % eval_period == 0 and local_rank <= 0:
                if prior_tune:
                    keys = ["lm_head.priors", "lm_head.gamma"]
                    if prompt_tune:
                        keys.append("transformer.wte.new_embed.weight")
                    model_state_dict = {key: model.state_dict()[key if local_rank<0 else "module."+key].cpu() for key in keys}
                elif prompt_tune:
                    keys = ["transformer.wte.new_embed.weight"]
                    model_state_dict = {key: model.state_dict()[key if local_rank<0 else "module."+key].cpu() for key in keys}
                elif head_tune:
                    keys = ["lm_head.my_lm_head.weight"]
                    model_state_dict = {key: model.state_dict()[key if local_rank<0 else "module."+key].cpu() for key in keys}
                elif transform_tune:
                    keys = ["lm_head.transform.weight"]
                    model_state_dict = {key: model.state_dict()[key if local_rank<0 else "module."+key].cpu() for key in keys}
                else:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                torch.save(model_state_dict,
                           os.path.join(output_dir, "model-{}.pt".format(global_step)))
                logger.info("Saving model at global_step=%d (train loss %.2f)" % \
                            (global_step, np.mean(train_losses)))
                train_losses = []

            scaler.update()

            if global_step==num_training_steps:
                break

        if global_step==num_training_steps:
            break

    logger.info("Finish training")

def inference(model, inputs, batch_size, prior_inputs=None, return_logits=False, bad=False):
    dataloader = get_dataloader(inputs, batch_size, is_training=False, prior_inputs=prior_inputs)

    all_losses = []
    for batch in tqdm(dataloader):
        input_ids=batch[0].cuda()
        attention_mask=batch[1].cuda()
        token_type_ids=batch[2].cuda()
        classes=batch[3].cuda()
        prior_input_ids=None
        prior_attention_mask=None
        prior_token_type_ids=None

        if len(batch)==4:
            labels=None
        elif len(batch)==7:
            labels=None
            prior_input_ids=batch[4].cuda()
            prior_attention_mask=batch[5].cuda()
            prior_token_type_ids=batch[6].cuda()
        elif len(batch)==5:
            labels=batch[4].cuda()
        else:
            labels=batch[7]

        with torch.no_grad():
            loss = run_model(model, input_ids, attention_mask, token_type_ids, regularization_weight=0,
                             prior_input_ids=prior_input_ids, prior_attention_mask=prior_attention_mask,
                             prior_token_type_ids=prior_token_type_ids, classes=classes, labels=labels,
                             return_logits=return_logits, bad=bad, local_rank=-1)

        all_losses += loss.cpu().detach().numpy().tolist()

    return all_losses


def run_model(model, input_ids, attention_mask, token_type_ids, regularization_weight,
              prior_input_ids=None, prior_attention_mask=None, prior_token_type_ids=None,
              aux_weight=0.0, target_indices=None, classes=None, labels=None, return_logits=False, 
              bad=False, local_rank=-1):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()

    if return_logits:
        softmax = torch.nn.Softmax(dim=-1)
        return -torch.log(softmax(logits))

    if labels is None:
        labels = input_ids
    labels = labels[..., 1:].contiguous()
    label_mask = token_type_ids[..., 1:].contiguous()


    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    prior_values = 0
    if (hasattr(model.lm_head, 'priors') if local_rank<0 else hasattr(model.module.lm_head, 'priors')):
        priors = model.lm_head.priors if local_rank<0 else model.module.lm_head.priors
        gamma = model.lm_head.gamma if local_rank<0 else model.module.lm_head.gamma
        prior_values = torch.abs(gamma) * loss_fct(priors.expand(len(classes), len(priors)), classes) + regularization_weight * (torch.linalg.norm(priors) ** 2)
    elif prior_input_ids != None:
        prior_outputs = model(input_ids=prior_input_ids, attention_mask=prior_attention_mask)
        prior_logits = outputs.logits[..., :-1, :].contiguous()

        prior_labels = prior_input_ids
        prior_labels = prior_labels[..., 1:].contiguous()
        prior_label_mask = prior_token_type_ids[..., 1:].contiguous()
        prior_losses = loss_fct(prior_logits.view(-1, prior_logits.size(-1)),
                                prior_labels.view(-1))
        prior_losses = prior_losses.view(prior_logits.size(0), prior_logits.size(1)) * prior_label_mask
        prior_values = torch.sum(prior_losses, axis=1) / torch.sum(prior_label_mask, axis=1)

    aux_loss = 0.0
    if target_indices != None:
        layer = model.transformer.wte if local_rank < 0 else model.module.transformer.wte
        aux_loss = aux_weight * (torch.sum((layer.embed.state_dict()["weight"][target_indices] - layer.new_embed.weight) ** 2))
        # aux_loss = aux_weight * torch.linalg.norm(layer.embed.state_dict()["weight"][target_indices] - layer.new_embed.weight, ord=1, dim=(0,1))

    losses = loss_fct(logits.view(-1, logits.size(-1)),
                      labels.view(-1)) # [batch_size, length]
    losses = losses.view(logits.size(0), logits.size(1)) * label_mask 
    return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1) * (-1 if bad else 1) + prior_values + aux_loss

