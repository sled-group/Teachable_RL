import numpy as np
import torch
import utils
import re
import json
import openai

from decision_transformer.models.sentencebert import sentencebert_encode
from homegrid.prompt_template import PROMPT_ENABLE_NO_FEEDBACK_RQ1, PROMPT_ENABLE_NO_FEEDBACK_RQ2

# Fill in openai api key for online gpt access
openai.api_key = ""

ACTION_MAP = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "pickup",
    5: "drop",
    6: "get",
    7: "pedal",
    8: "grasp",
    9: "lift",
}

np.random.seed(42)

def evaluate_episode_rtg(
    state_dim,
    act_dim,
    model,
    max_ep_len=100,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    seed=None,
    informativeness=None,
    diversity=None,
    train_ratio=0.8,
    val_ratio=0.1,
    lang_mode="test",
    rq=None,
    disturb_hind=False,
    disturb_fore=False,
):
    encoder = sentencebert_encode

    valid = True

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    env = utils.make_env(
        need_reset=False,
        gpt_pool=(diversity in ["gpt_pool", "online_gpt"]),
        train_ratio=train_ratio,
        mode=lang_mode,
        val_ratio=val_ratio,
        disturb_hind=disturb_hind,
        disturb_fore=disturb_fore,
    )
    state, _ = env.reset(seed=seed)

    task_name = env.task
    task = encoder([task_name]).to(device=device, dtype=torch.float32)

    info = env.info["init_state_info"]

    state = state["image"]
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim[0], state_dim[1], state_dim[2])
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    languages = (encoder([env.task])).to(device=device, dtype=torch.float32)
    languages = languages.reshape(1, -1).to(device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            task.to(dtype=torch.float32),
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            languages.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        action = action.detach().cpu().numpy()

        action = torch.argmax(torch.nn.Sigmoid()(torch.tensor(action)))

        action = np.array(action, dtype=int)
        action_one_hot = torch.zeros(act_dim, device=device)
        action_one_hot[action] = 1
        actions[-1] = action_one_hot
        action = int(action)

        state, reward, done, _, info = env.step(action)
        action_failed_reason = info["action_status"]["action_failed_reason"]

        if diversity == "online_gpt":
            if rq == 1:
                prompt = (
                    PROMPT_ENABLE_NO_FEEDBACK_RQ2.replace(
                        "{if_carry_plate}", info["if_carry_plate"]
                    )
                    .replace("{bin_status}", info["bin_status"])
                    .replace("{robot_action}", ACTION_MAP[action])
                    .replace("{hindsight}", action_failed_reason)
                    .replace("{future_feedback}", state["log_language_info"])
                )
            else:
                prompt = (
                    PROMPT_ENABLE_NO_FEEDBACK_RQ1.replace("{task_name}", task_name)
                    .replace("{robot_action}", ACTION_MAP[action])
                    .replace("{hindsight}", action_failed_reason)
                    .replace("{future_feedback}", state["log_language_info"])
                )

            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                response = response.choices[0].message.content

                pattern = r"\{([^{}]*)\}"
                matches = re.findall(pattern, response)
                response = "{" + matches[0] + "}"

                response = json.loads(response)

                lang = response["response"]

                if not response["if_give_response"]:
                    lang = ""

            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"error happens at {seed}")
                print(prompt)
                print(response)
                return 0, 0, False
        else:
            if informativeness == "no_lang":
                lang = ""
            elif informativeness == "f":
                lang = state["log_language_info"]
            elif informativeness == "h":
                lang = action_failed_reason
            else:
                lang = action_failed_reason + " " + state["log_language_info"]
        
        lang = encoder([lang])
        state = state["image"]

        cur_state = (
            torch.from_numpy(state)
            .to(device=device)
            .reshape(1, state_dim[0], state_dim[1], state_dim[2])
        )
        cur_lang = lang.to(device=device).reshape(1, 768)
        states = torch.cat([states, cur_state], dim=0)
        languages = torch.cat([languages, cur_lang], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)

        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        if done:
            if t == 0:
                valid = False
            break

    return episode_return, episode_length, valid
