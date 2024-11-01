import argparse
import torch
import numpy as np
import pickle
import os
import torch.backends.cudnn as cudnn

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg

cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def main(variant):
    num_eval_episodes = variant["num_eval_episodes"]
    state_dim = (96, 96, 3)
    act_dim = 10
    lang_dim = 768
    max_ep_len = 100
    scale = 1000.0
    target_rew = 1.5
    if variant["gpu"] is None:
        device = variant.get("device", "cuda")
    else:
        device = f"cuda:{variant['gpu']}"

    K = variant["K"]

    # load state mean and state std
    if os.path.exists(f"{variant['load_model_path']}/state_mean_std.pkl"):
        with open(f"{variant['load_model_path']}/state_mean_std.pkl", "rb") as f:
            print(
                f"load {variant['load_model_path']}/state_mean_std.pkl for state mean and std"
            )
            state_mean, state_std = pickle.load(f)
    else:
        raise ValueError("state mean and std not found")
    
    with open("./data/seed_len.pkl", "rb") as f:
        seed_len = pickle.load(f)

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

    print(f"load model from folder {variant['load_model_path']}")
    if variant["model_iter"] is None:
        model.load_state_dict(
            torch.load(
                f"{variant['load_model_path']}/ckpt.pt",
                map_location=device,
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                f"{variant['load_model_path']}/model-{variant['model_iter']}.pt",
                map_location=device,
            )
        )

    if variant["rq"] == 1:
        seed_list = seed_len['rq1']['test']['seed']
        seed_testing_expert_len = seed_len['rq1']['test']['expert_len']
    else:
        seed_list = seed_len['rq2']['test']['seed']
        seed_testing_expert_len = seed_len['rq2']['test']['expert_len']
    
    returns, lengths, normalized_returns = [], [], []
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
            normalized_return = ret * seed_testing_expert_len[i] / max(length, seed_testing_expert_len[i])

            if not valid:
                continue
            returns.append(ret)
            lengths.append(length)
            normalized_returns.append(normalized_return)
            if (i + 1) % 10 == 0:
                print(f"episode {i + 1}, return {np.mean(returns)}, length {np.mean(lengths)}")

    result = {
        f"target_{target_rew}_return_mean": np.mean(returns),
        f"target_{target_rew}_length_mean": np.mean(lengths),
        f"target_{target_rew}_normalized_return_mean": np.mean(normalized_returns),
    }

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--model_iter", type=int, default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("-i", "--informativeness", type=str, choices=["no_lang", "f", "h", "h_f"], default="h_f")
    parser.add_argument("-d", "--diversity", type=str, choices=["template", "gpt_pool", "online_gpt"], default="online_gpt")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lang_mode", type=str, default="test")
    parser.add_argument("--rq", type=int, choices=[1, 2], default=None)

    args = parser.parse_args()

    assert args.load_model_path is not None
    assert not (args.informativeness == "no_lang" and args.diversity in ["gpt_pool", "online_gpt"])
    assert not (args.informativeness in ["f", "h"] and args.diversity == "online_gpt")

    main(variant=vars(args))
