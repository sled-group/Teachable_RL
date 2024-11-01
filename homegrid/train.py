import numpy as np
import torch
import wandb
import argparse
import pickle
import random
import os
import torch.backends.cudnn as cudnn

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def experiment(variant):
    if variant["gpu"] is None:
        device = variant.get("device", "cuda")
    else:
        device = f"cuda:{variant['gpu']}"
    log_to_wandb = variant.get("log_to_wandb", False)

    env_name = variant["env"]

    group_name = f"{env_name}-rq{variant['rq']}"
    
    if variant["load_model_path"] is None:
        group_name += "-scratch"
    else:
        group_name += "-adapt"
        
    group_name += f"-{variant['diversity']}"
    
    group_name += f"-{variant['informativeness']}"
    
    if variant["shot"] is not None:
        group_name += f"-{variant['shot']}shots"

    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    rng = np.random.RandomState(42)
    
    max_ep_len = 100

    env_targets = [1.5]
    scale = 1000.0

    state_dim = (96, 96, 3)
    act_dim = 10
    lang_dim = 768

    # load dataset
    if variant["load_data_path"]:
        dataset_path = variant["load_data_path"]
    else:
        raise AssertionError("load_data_path is not provided")

    with open(dataset_path, "rb") as f:
        print("load", dataset_path)
        trajectories = pickle.load(f)

    with open("./data/seed_len.pkl", "rb") as f:
        seed_len = pickle.load(f)
    
    # save all path information into separate lists
    states, lan_embeds, traj_lens, returns = [], [], [], []
    for path in trajectories:
        states.append(path["observations"])
        lan_embeds.append(path["languages"])
        traj_lens.append(len(path["observations"]))
        returns.append(sum(path["rewards"]))

    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)

    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    if variant["save"]:
        print(f"save state mean and std to {variant['save']}/state_mean_std.pkl")

        if not os.path.exists(f"{variant['save']}"):
            os.makedirs(f"{variant['save']}")

        with open(f"{variant['save']}/state_mean_std.pkl", "wb") as f:
            pickle.dump([state_mean, state_std], f)

    num_timesteps = sum(traj_lens)

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = rng.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, l, r, d, rtg, timesteps, mask, tasks = [], [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            tasks.append(traj["task_name"].reshape(1, 1, 768))
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(
                traj["observations"][si : si + max_len].reshape(
                    1, -1, state_dim[0], state_dim[1], state_dim[2]
                )
            )
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))

            l.append(traj["languages"][si : si + max_len].reshape(1, -1, 768))

            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [
                    np.zeros(
                        (1, max_len - tlen, state_dim[0], state_dim[1], state_dim[2])
                    ),
                    s[-1],
                ],
                axis=1,
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            l[-1] = np.concatenate([np.zeros((1, max_len - tlen, 768)), l[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )
        tasks = torch.from_numpy(np.concatenate(tasks, axis=0)).to(
            dtype=torch.float32, device=device
        )
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        l = torch.from_numpy(np.concatenate(l, axis=0)).to(
            dtype=torch.float32, device=device
        )

        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return tasks, s, l, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths, normalized_returns = [], [], []
            if variant["rq"] == 1:
                seed_list = seed_len['rq1']['validation']['seed']
                seed_validation_expert_len = seed_len['rq1']['validation']['expert_len']
            else:
                seed_list = seed_len['rq2']['validation']['seed']
                seed_validation_expert_len = seed_len['rq2']['validation']['expert_len']

            for i in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length, valid = evaluate_episode_rtg(
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        seed=int(seed_list[i]),
                        informativeness=variant["informativeness"],
                        diversity=variant["diversity"],
                        train_ratio=variant["train_ratio"],
                        lang_mode=variant["lang_mode"],
                        val_ratio=variant["val_ratio"],
                        rq=variant["rq"],
                    )
                    normalized_return = ret * seed_validation_expert_len[i] / max(length, seed_validation_expert_len[i])
                if not valid:
                    continue
                returns.append(ret)
                lengths.append(length)
                normalized_returns.append(normalized_return)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_normalized_return_mean": np.mean(normalized_returns),
            }

        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        lang_dim=lang_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
    )

    model = model.to(device=device)

    if variant["load_model_path"] is not None:
        print("load model in folder:", variant["load_model_path"])
        if variant["model_iter"] is None:
            print("load checkpoint")
            model.load_state_dict(torch.load(f"{variant['load_model_path']}/ckpt.pt"))
        else:
            model.load_state_dict(
                torch.load(f"{variant['load_model_path']}/model-{variant['model_iter']}.pt")
            )
    else:
        print("train from scratch!")

    warmup_steps = variant["warmup_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.BCEWithLogitsLoss()(
            a_hat, a
        ),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    if log_to_wandb:
        wandb.login()
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )
        # wandb.watch(model)  # wandb has some bug
    print("initial eval")
    outputs = trainer.initial_eval()
    if log_to_wandb:
        wandb.log(outputs)

    print("in training")
    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        if log_to_wandb:
            wandb.log(outputs)
        if variant["save"] is not None:
            print(f"save model at {variant['save']}/model-{iter}.pt")
            torch.save(model.state_dict(), f"{variant['save']}/model-{iter}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="homegrid")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=30)
    parser.add_argument("--num_steps_per_iter", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", type=int, default=1)
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--model_iter", type=int, default=29)
    parser.add_argument("-i", "--informativeness", type=str, choices=["no_lang", "f", "h", "h_f"], default="h_f")
    parser.add_argument("-d", "--diversity", type=str, choices=["template", "gpt_pool"], default="gpt_pool")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lang_mode", type=str, default="val")
    parser.add_argument("--load_data_path", type=str, default=None)
    parser.add_argument("--rq", type=int, choices=[1, 2], default=None)

    args = parser.parse_args()

    experiment(variant=vars(args))
