import os
import argparse
import pickle as pkl
import random
import torch
import math
import logging
import numpy as np

from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from data import load_data, prepare_data, load_prompt, output_metrices
from run import train, inference
from model_util import load_checkpoint, set_extra_embeddings, \
    set_separate_lm_head, set_separate_embeddings, set_transformed_lm_head, set_prior
from util import get_prompts, get_paths, flatten_label_losses, \
    prepend_task_tokens, reassign_output_tokens, f1_score
from templates import TEMPLATES

N_LABELS_DICT = {"SST-2": 2, "sst-5": 5, "mr": 2, "cr": 2, "mpqa": 2,
                 "subj": 2, "trec": 6, "trec-5": 5, "trec-4": 4, "trec-3": 3, "CoLA": 2,
                 "amazon": 5, "yelp_full": 5, "yelp_binary": 2,
                 "agnews": 4, "copa": 2, "boolq": 2,
                 "RTE": 2, "cb": 3,
                 "yahoo": 10, "dbpedia": 14, 'climate_fever': 4, 
                 'ethos-national_origin': 2, 'ethos-race': 2,
                 'ethos-religion': 2, 'financial_phrasebank': 3, 
                 'hate_speech18': 2, 'medical_questions_pairs': 2, 
                 'poem_sentiment': 4, 'superglue-cb': 3, 
                 'tweet_eval-hate': 2, 'tweet_eval-stance_atheism': 3, 
                 'tweet_eval-stance_feminist': 3, 'anli': 3, 
                 'glue-mnli': 3, 'glue-qnli': 2, 'glue-rte': 2, 
                 'glue-wnli': 2, 'scitail': 2, 'sick': 3,
                 'ai2_arc': 4, 'codah': 4, 'commonsense_qa': 5, 
                 'openbookqa': 4, 'qasc': 8, 'quarel': 2, 'quartz-no_knowledge': 2, 
                 'quartz-with_knowledge': 2, 'superglue-copa': 2, 'wino_grande': 2
}


def main(logger, args):
    args.gpt2 = args.gpt2.replace("gpt2-small", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    model = None

    if args.train_task is None:
        # standard case where the training task and the test task are the same
        train_task = args.task
    else:
        # zero-shot transfer case where the training task is different from the test task
        train_task = args.train_task
        # assert args.do_check

    # datasets where the average input length is long
    long_datasets = ["cr", "subj", "agnews",
                     "amazon", "yelp_full", "yelp_binary", "boolq",
                     "dbpedia", "yahoo", 'climate_fever', 
                     'ethos-national_origin', 'ethos-race', 
                     'ethos-religion', 'financial_phrasebank', 'hate_speech18', 
                     'medical_questions_pairs', 'superglue-cb', 
                     'tweet_eval-hate', 'anli', 'glue-mnli', 'glue-qnli', 
                     'glue-rte', 'glue-wnli', 'scitail', "ai2_arc", "codah", "openbookqa",
                     'quarel', 'quartz-with_knowledge']
    max_length = 256 if train_task in long_datasets else 128
    batch_size = int(args.batch_size / 2) if train_task in long_datasets else args.batch_size

    logger.info("%s %s" % (args.method, args.task))

    assert args.method in ["direct", "channel"]

    if args.use_demonstrations:
        assert args.do_zeroshot and not args.do_train

    if args.ensemble:
        assert args.use_demonstrations

    if args.checkpoint_dir is not None:
        assert args.do_train

    if args.do_train or args.use_demonstrations:
        assert args.train_seed > 0

    if args.prompt_task is not None:
        assert args.prompt_tune and args.init_method == "manual"
        prompt = load_prompt(args.prompts_dir, args.prompt_task, int(args.prompt_file_len))
        logger.info("Using prompt: %s" % prompt)
    else:
        prompt = None

    n_templates = 1
    k = int(args.k)
    seed = int(args.seed)
    local_rank = int(args.local_rank) if args.local_rank is not None else -1

    if args.use_tau:
        assert N_LABELS_DICT[args.task] == 2

    train_data = load_data(args.data_dir, train_task, k, seed, "train")
    if args.split is None:
        assert args.do_zeroshot
        dev_data = None
    else:
        dev_data = load_data(args.data_dir, args.task, k, seed, args.split)

    
    if local_rank >= 0:
        torch.distributed.init_process_group("nccl", init_method='env://')
    
    if not args.robust_eval:
        accs, f1s = [], []
        # run over different templates
        for template_idx in range(n_templates):
            acc, f1 = run(logger, args.do_train, args.do_zeroshot, args.use_tau,
                    args.task, train_task, args.prompt_task,
                    k, seed, args.train_seed,
                    args.out_dir, args.checkpoint_dir, args.split,
                    tokenizer, model, train_data, dev_data,
                    batch_size, max_length, args.gpt2, args.init_method, args.prefix_type,
                    template_idx, args.method,
                    args.lr, args.prior_weight, args.aux_weight, args.regularization_weight,
                    args.warmup_steps, args.num_training_steps, args.eval_period,
                    args.robust_eval, local_rank, prompt,
                    use_demonstrations=args.use_demonstrations,
                    use_calibration=args.use_calibration,
                    ensemble=args.ensemble,
                    is_null=args.split is None,
                    prompt_tune=args.prompt_tune,
                    head_tune=args.head_tune,
                    transform_tune=args.transform_tune,
                    prior_tune=args.prior_tune,
                    bad=args.bad,
                    do_check=args.do_check,
                    n_prefix=args.n_prefix)

            accs.append(acc)
            f1s.append(f1)

        if args.split is not None:
            logger.info("Accuracy = %.1f (Avg) / %.1f (Worst)" % (100*np.mean(accs), 100*np.min(accs)))
            logger.info("Micro-F1 = %.1f (Avg) / %.1f (Worst)" % (100*np.mean(f1s), 100*np.min(f1s)))
    
    else:
        assert args.prompt_tune and args.prompt_task != None and train_task == args.task

        tseeds = [1, 10, 100]
        seed_results = []
        for tseed in tseeds:
            acc, f1, mapped_prompt, norm_distance, mapped_acc, mapped_f1 = run(logger, args.do_train, args.do_zeroshot, args.use_tau,
                            args.task, train_task, args.prompt_task,
                            k, seed, tseed,
                            args.out_dir, args.checkpoint_dir, args.split,
                            tokenizer, model, train_data, dev_data,
                            batch_size, max_length, args.gpt2, args.init_method, args.prefix_type,
                            0, args.method,
                            0.01, args.prior_weight, args.aux_weight, args.regularization_weight,
                            args.warmup_steps, args.num_training_steps, args.eval_period,
                            args.robust_eval, local_rank, prompt,
                            use_demonstrations=args.use_demonstrations,
                            use_calibration=args.use_calibration,
                            ensemble=args.ensemble,
                            is_null=args.split is None,
                            prompt_tune=args.prompt_tune,
                            head_tune=args.head_tune,
                            transform_tune=args.transform_tune,
                            prior_tune=args.prior_tune,
                            bad=args.bad,
                            do_check=args.do_check,
                            n_prefix=args.n_prefix,
                            f1_threshold=args.f1_threshold,
                            prompt_file_len=args.prompt_file_len)
            seed_results.append({
                "soft_accuracy": acc,
                "prompt_f1": f1_score(mapped_prompt, prompt),
                "soft_macro-f1": f1,
                "mapped_prompt": mapped_prompt,
                "norm_distance": norm_distance,
                "mapped_accuracy": mapped_acc,
                "mapped_macro-f1": mapped_f1 
            })
        test_result = {
            "soft_accuracy": np.average([seed_result["soft_accuracy"] for seed_result in seed_results]),
            "prompt_f1": np.average([seed_result["prompt_f1"] for seed_result in seed_results]),
            "soft_macro-f1": np.average([seed_result["soft_macro-f1"] for seed_result in seed_results]),
            "norm_distance": np.average([seed_result["norm_distance"] for seed_result in seed_results]),
            "mapped_accuracy": np.average([seed_result["mapped_accuracy"] for seed_result in seed_results]),
            "mapped_macro-f1": np.average([seed_result["mapped_macro-f1"] for seed_result in seed_results]) 
        }
        # logger.info("Results for robust evalution on {} with prompt of {} with lr={}, gamma={}".format(args.task, args.prompt_task, best_lr, best_gamma))
        output_metrices(args, seed_results, test_result, prompt, len(tokenizer(prompt)["input_ids"]))
        

def run(logger, do_train, do_zeroshot, use_tau, task, train_task, prompt_task,
        k, seed, train_seed,
        out_dir, checkpoint_dir, split, tokenizer, model,
        train_data, dev_data,
        batch_size, max_length, gpt2, init_method, prefix_type,
        template_idx, method_type, learning_rate, 
        prior_weight, aux_weight, regularization_weight,
        warmup_steps, num_training_steps, eval_period,
        robust_eval, local_rank, prompt,
        use_demonstrations=False,
        use_calibration=False,
        ensemble=False,
        is_null=False,
        prompt_tune=False,
        head_tune=False,
        transform_tune=False,
        prior_tune=False,
        bad=False,
        do_check=False, n_prefix=-1,
        f1_threshold=0.95, prompt_file_len=-1):

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(train_seed)

    if head_tune or transform_tune:
        assert method_type == "direct"

    if init_method == "manual":
        n_prefix = len(tokenizer(prompt)["input_ids"]) if n_prefix < 0 else n_prefix
    elif init_method == "vocab":
        n_prefix = 20

    n_classes = N_LABELS_DICT.get(task, None)
    templates = get_prompts(task, template_idx)

    n_classes_train = N_LABELS_DICT.get(train_task, None)
    templates_train = get_prompts(train_task, template_idx)

    # TODO: maybe need to consider this for the newly added datasets
    if task in ["yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
        templates = [t.replace(".", " .") for t in templates]

    max_length_per_example = max_length

    if use_demonstrations and not ensemble:
        assert do_zeroshot and not do_train
        mem = batch_size * max_length
        if n_classes == 2:
            max_length = max_length * k
        elif n_classes in [4, 5]:
            max_length = int(max_length * 1.5 * k)
        elif n_classes in [6]:
            max_length = int(max_length * 2 * k)
        else:
            max_length = 1024

        max_length = min(max_length, 1024)
        batch_size = int(mem / max_length)

    if checkpoint_dir is not None:
        prompt_tuned = prior_tune
        prior_tuned = prompt_tune
        prior_tune = True
        prompt_tune = True

    if do_zeroshot:
        cache_paths = [get_paths(out_dir, gpt2, method_type, task, do_zeroshot,
                                 k, seed, train_seed, split, template_idx,
                                 use_demonstrations=use_demonstrations,
                                 ensemble=ensemble)]
        checkpoints = [None]

    else:
        out_dir = get_paths(out_dir, gpt2, method_type, train_task, do_zeroshot,
                            k, seed, train_seed, split, template_idx,
                            batch_size, learning_rate, warmup_steps,
                            regularization_weight, prior_weight, aux_weight, 
                            init_method, prompt_task,
                            use_demonstrations=use_demonstrations,
                            ensemble=ensemble,
                            bad=bad,
                            prompt_tune=prompt_tune,
                            head_tune=head_tune,
                            transform_tune=transform_tune,
                            prior_tune=prior_tune,
                            n_prefix=n_prefix,
                            f1_threshold=f1_threshold,
                            prompt_file_len=prompt_file_len)

        k = int(k)
        # eval_period = 500
        # if k == 16384:
        #     num_training_steps = 1000
        # elif k == -1:
        #     num_training_steps = 2000
        # else:
        #     num_training_steps = 400

        cache_paths = [os.path.join(out_dir, "{}cache-{}-{}-{}.pkl".format(
            task + "-" if train_task != task else "",
            split, step, prefix_type))
                       for step in range(num_training_steps, 0, -eval_period)]
        checkpoints = [os.path.join(out_dir, "model-{}.pt".format(step))
                       for step in range(num_training_steps, 0, -eval_period)]

    mapping = None

    mc_datasets = ['ai2_arc', 'codah', 'commonsense_qa', 'cosmos_qa', 'dream', 
                   'hellaswag', 'openbookqa', 'qasc', 'quail', 'quarel', 'quartz-no_knowledge', 
                   'quartz-with_knowledge', 'race-high', 'race-middle', 'sciq', 'social_i_qa', 
                   'superglue-copa', 'swag', 'wino_grande', 'wiqa']


    if do_train and (head_tune or not do_check):

        inputs = prepare_data(
            tokenizer, None, train_data,
            max_length=max_length,
            max_length_per_example=max_length_per_example,
            n_classes=n_classes_train,
            templates=templates_train,
            method_type=method_type,
            is_training=True,
            ensemble=ensemble)

        if train_task in mc_datasets and prior_tune:
            prior_train_data = [(TEMPLATES[train_task][template_idx][1], label, choices) for sent, label, choices in train_data]
            prior_input_tensors = prepare_data(
                tokenizer, None, prior_train_data,
                max_length=max_length,
                max_length_per_example=max_length_per_example,
                n_classes=n_classes_train,
                templates=None,
                method_type="direct",
                is_training=True,
                ensemble=ensemble)
        else:
            prior_input_tensors = None

        logger.info(out_dir)

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        if not do_check and not np.all([os.path.exists(checkpoint) for checkpoint in checkpoints]):

            if checkpoint_dir is not None:
                if not os.path.exists(checkpoint_dir):
                    logger.info("checkpoint %s not found..." % checkpoint_dir)
                    assert False
                else:
                    logger.info("Loading the checkpoint")
                    torch.cuda.empty_cache()
                    del model
                    model = load_checkpoint(gpt2, checkpoint_dir,
                                            prompt_tune=prompt_tuned,
                                            head_tune=head_tune,
                                            transform_tune=transform_tune,
                                            prior_tune=prior_tuned and train_task not in mc_datasets,
                                            n_prefix=n_prefix,
                                            mapping=mapping,
                                            n_classes=n_classes)

                    if prompt_tuned:
                        for param in model.parameters():
                            param.requires_grad = False
                        if prior_weight < 0:
                            set_prior(model, n_classes, 1.0)
                            model.lm_head.gamma.requires_grad = True
                        else:
                            set_prior(model, n_classes, prior_weight)
                            model.lm_head.gamma.requires_grad = False
                        model.lm_head.priors.requires_grad = True 
                    elif prior_tuned:
                        for param in model.parameters():
                            param.requires_grad = False

                        set_extra_embeddings(model, n_prefix)
                        inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

                    # For debugging
                    if prior_tune and train_task not in mc_datasets:
                        logger.info("The prior's value are: {}".format(model.lm_head.priors.tolist()))
                        logger.info("The weight for priors is: {}".format(model.lm_head.gamma.item()))

                    model = model.cuda()
                    logger.info("Finished loading the checkpoint")
            else:
                model = GPT2LMHeadModel.from_pretrained(gpt2)

                if prior_tune and train_task not in mc_datasets:
                    for param in model.parameters():
                        param.requires_grad = False
                    if prior_weight < 0:
                        set_prior(model, n_classes, 1.0)
                        model.lm_head.gamma.requires_grad = True
                    else:
                        set_prior(model, n_classes, prior_weight)
                        model.lm_head.gamma.requires_grad = False
                    model.lm_head.priors.requires_grad = True 
                    if prompt_tune:
                        set_extra_embeddings(model, n_prefix)
                        inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

                elif prompt_tune:
                    for param in model.parameters():
                        param.requires_grad = False

                    if init_method == "manual":
                        prompt_ids = tokenizer(prompt)["input_ids"]
                        set_extra_embeddings(model, n_prefix, prompt_ids)
                        logger.info("Using a prompt of size {}".format(len(prompt_ids)))
                    else:
                        set_extra_embeddings(model, n_prefix, init_method)
                    inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

                elif head_tune:
                    mapping, inputs = reassign_output_tokens(inputs, for_labels=True)
                    logger.info("Created mapping with {} vocabs".format(len(mapping)))
                    set_separate_lm_head(model, mapping)
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.lm_head.my_lm_head.parameters():
                        param.requires_grad = True

                elif transform_tune:
                    set_transformed_lm_head(model)
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.lm_head.transform.parameters():
                        param.requires_grad = True

                model = model.cuda()

            # if torch.cuda.device_count() > 1:
            #     model = torch.nn.DataParallel(model) 
            
            # distributed
            if local_rank >= 0:
            #     torch.distributed.init_process_group("nccl", init_method='env://')
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

            train(logger, model, inputs, batch_size, out_dir, local_rank,
                  prior_inputs=prior_input_tensors,
                  learning_rate=learning_rate,
                  regularization_weight=regularization_weight,
                  aux_weight=aux_weight,
                  target_indices=tokenizer(prompt)["input_ids"] if prompt != None else None,
                  warmup_steps=warmup_steps,
                  eval_period=eval_period,
                  num_training_steps=num_training_steps,
                  prompt_tune=prompt_tune,
                  head_tune=head_tune,
                  transform_tune=transform_tune,
                  prior_tune=prior_tune,
                  bad=bad)

    # if local_rank > 0:
    #     logger.info("rank {} exited!".format(local_rank))
    #     exit()

    input_tensors = prepare_data(
        tokenizer, train_data, dev_data,
        max_length=max_length,
        max_length_per_example=max_length_per_example,
        n_classes=n_classes,
        templates=templates,
        method_type=method_type,
        use_demonstrations=use_demonstrations,
        ensemble=ensemble,
        is_null=is_null)

    if train_task in mc_datasets and prior_tune:
        prior_dev_data = [(TEMPLATES[train_task][template_idx][1], label, choices) for sent, label, choices in dev_data]
        prior_input_tensors = prepare_data(
            tokenizer, None, prior_dev_data,
            max_length=max_length,
            max_length_per_example=max_length_per_example,
            n_classes=n_classes_train,
            templates=None,
            method_type="direct",
            ensemble=ensemble)
    else:
        prior_input_tensors = None

    if prompt_tune:
        input_tensors = prepend_task_tokens(tokenizer, input_tensors, n_prefix)

    if head_tune:
        # some tricks in case train_task and test_task are different
        # TODO: maybe need to consider this for the newly added datasets
        if task != train_task:
            if task in ["sst-5", "yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
                input_tensors = [input_tensors[0], input_tensors[-1]]
                if head_tune:
                    label_counter = {'0': '0', '4': '1'}
                    dev_data = [(x, label_counter.get(y, '-1')) for x, y in dev_data]
            elif task in ["SST-2", "mr"] and train_task in ["SST-2", "mr", "sst-5"]:
                pass
            else:
                raise NotImplementedError()

        if mapping is None:
            mapping, inputs = reassign_output_tokens(inputs, for_labels=head_tune)

        train_labels = set([label for _, label in train_data])
        if len(train_labels) != n_classes:
            train_labels = sorted(train_labels)
            input_tensors = [input_tensors[int(l)] for l in train_labels]
            dev_data = [(sent, str(train_labels.index(l)) if l in train_labels else -1)
                        for sent, l in dev_data]

        _, input_tensors = reassign_output_tokens(input_tensors, for_labels=head_tune,
                                                  mapping={v: k for k, v in mapping.items()})
        logger.info(mapping)
        logger.info("Checked that train mapping and test mapping are identical")


    # for debugging ...
    logger.info("Checking the first example...")
    input_ids = input_tensors[0]["input_ids"][0].numpy().tolist()
    token_type_ids = input_tensors[0]["token_type_ids"][0].numpy().tolist()
    logger.info("Input:")
    logger.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
    logger.info("Output:")
    logger.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))
    if prior_tune and task in mc_datasets:
        input_ids = prior_input_tensors[0]["input_ids"][0].numpy().tolist()
        token_type_ids = prior_input_tensors[0]["token_type_ids"][0].numpy().tolist()
        logger.info("Prior input:")
        logger.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
        logger.info("Prior output:")
        logger.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))

    results = []
    for cache_path, checkpoint in zip(cache_paths, checkpoints):

        logger.info(cache_path)

        # if there is a cache, load it
        if False: # os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                losses = pkl.load(f)
        else:
            if checkpoint is not None and not os.path.exists(checkpoint):
                logger.info("checkpoint %s not found..." % checkpoint)
                assert False

            if checkpoint is None and model is not None and do_zeroshot:
                logger.info("Reusing the loaded model...")
                pass
            else:
                logger.info("Loading the model")
                torch.cuda.empty_cache()
                del model
                model = load_checkpoint(gpt2, checkpoint,
                                        prompt_tune=prompt_tune,
                                        head_tune=head_tune,
                                        transform_tune=transform_tune,
                                        prior_tune=prior_tune and task not in mc_datasets,
                                        n_prefix=n_prefix,
                                        mapping=mapping,
                                        n_classes=n_classes)
                
                if prefix_type == "discrete":
                    prefix_ids, aux_loss = model.transformer.wte.map_to_discrete()
                    logger.info("The mapped discrete prefix is: {}".format(tokenizer.decode(prefix_ids)))
                    logger.info("The norm distance is: {}".format(aux_loss))

                # For debugging
                if prior_tune and task not in mc_datasets:
                    logger.info("The prior's value are: {}".format(model.lm_head.priors.tolist()))
                    logger.info("The weight for priors is: {}".format(model.lm_head.gamma.item()))

                model = model.cuda()
                model.eval()
                logger.info("Finished loading the model")

            losses = []
            for i, input_tensor in enumerate(input_tensors):
                losses.append(inference(model,
                                        input_tensor,
                                        batch_size * 8,
                                        prior_inputs=prior_input_tensors[i] if prior_input_tensors != None else None,
                                        bad=bad))

            with open(cache_path, "wb") as f:
                pkl.dump(losses, f)

        if is_null:
            continue

        if ensemble:
            losses = flatten_label_losses(losses, dev_data)

        if use_calibration:
            bias_path = cache_path.replace(split, "None")
            assert os.path.exists(bias_path), bias_path
            with open(bias_path, "rb") as f:
                bias_losses = pkl.load(f)

            for i, (bias_loss, loss) in enumerate(zip(bias_losses, losses)):
                loss = np.array(loss)
                bias_loss = np.array(bias_loss)
                if ensemble:
                    bias_loss = bias_loss.reshape(1, -1)
                losses[i] = loss - bias_loss

        if task in mc_datasets:
            dev_data = [(sent, label) for sent, label, _ in dev_data]
        if not use_tau:
            acc, f1 = evaluate(dev_data, {str(i): loss for i, loss in enumerate(losses)})
        else:
            acc, f1, tau = float('-inf'), float('-inf'), float('-inf')
            for tau_cur in np.arange(-1.000, 1.000, 0.001):
                acc_cur, f1_cur = evaluate(dev_data, {str(i): loss for i, loss in enumerate(losses)}, tau=tau_cur)
                if f1_cur > f1:
                    acc, f1, tau = acc_cur, f1_cur, tau_cur
            logger.info("tau = {}".format(tau))
        logger.info(acc)
        logger.info(f1)

        if robust_eval:
            prefix_ids, aux_loss = model.transformer.wte.map_to_discrete()
            logger.info("The mapped discrete prefix is: {}".format(tokenizer.decode(prefix_ids)))
            logger.info("The norm distance is: {}".format(aux_loss))

            logger.info("Evaluating mapped discrete prompt")
            mapped_losses = []
            for i, input_tensor in enumerate(input_tensors):
                mapped_losses.append(inference(model,
                                        input_tensor,
                                        batch_size * 8,
                                        prior_inputs=prior_input_tensors[i] if prior_input_tensors != None else None,
                                        bad=bad))
            mapped_acc, mapped_f1 = evaluate(dev_data, {str(i): loss for i, loss in enumerate(mapped_losses)})
            logger.info(mapped_acc)
            logger.info(mapped_f1)

            return acc, f1, tokenizer.decode(prefix_ids), aux_loss.item(), mapped_acc, mapped_f1
        else:
            return acc, f1
    return None, None

def evaluate(dev_data, label_losses, tau=0, is_classification=True):
    if type(label_losses)==list:
        label_losses = np.array(label_losses)
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l:np.sum(label_losses[l][idx]) for l in label_losses}
        if math.isclose(tau, 0):
            prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        else:
            prediction = "0" if (np.exp(-label_loss["0"]) - np.exp(-label_loss["1"])) / (np.exp(-label_loss["0"]) + np.exp(-label_loss["1"])) > tau else "1"
        accs.append(prediction==label)
        precisions[prediction].append(prediction==label)
        recalls[label].append(prediction==label)

    if not is_classification:
        return np.mean(accs)

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return np.mean(accs), np.mean(f1s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--do_check", default=False, action="store_true")

    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_tau", default=False, action="store_true")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--prompt_tune", default=False, action="store_true")
    parser.add_argument("--head_tune", default=False, action="store_true")
    parser.add_argument("--transform_tune", default=False, action="store_true")
    parser.add_argument("--prior_tune", default=False, action="store_true")
    parser.add_argument("--bad", default=False, action="store_true")
    parser.add_argument("--robust_eval", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--train_task", type=str, default=None)
    parser.add_argument("--prompt_task", type=str, default=None)

    parser.add_argument("--k", type=str, default="16")
    parser.add_argument("--seed", type=str, default="100")
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--regularization_weight", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_training_steps", type=int, default=2000)
    parser.add_argument("--eval_period", type=int, default=500)
    parser.add_argument("--f1_threshold", type=float, default=0.95)
    parser.add_argument("--prompt_file_len", type=int, default=-1)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompts_dir", type=str, default="prompts")

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--n_prefix", type=int, default=-1)
    parser.add_argument("--gpt2", type=str, default="gpt2-large")
    parser.add_argument("--init_method", type=str, default="vocab")
    parser.add_argument("--prefix_type", type=str, default="soft")
    parser.add_argument("--ablation_type", type=str, default="gamma")

    parser.add_argument("--local_rank", type=str, default=None)

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
