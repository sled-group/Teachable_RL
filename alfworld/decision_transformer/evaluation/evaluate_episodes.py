import numpy as np
import torch
import json
import openai
import re

from decision_transformer.models.sentencebert import sentencebert_encode
from decision_transformer.evaluation.eval_utils import (
    extract_task_name, 
    parse_observation, 
    form_action_str, 
    get_subgoal_finished, 
    generate_gpt_feedback
)

from llfbench.prompt_template import PROMPT

# Fill in openai api key for online gpt access
openai.api_key = ""
            
rng = np.random.default_rng()

def evaluate_episode_rtg(
    env,
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
    lang_mode="val",
    disturb_hind=0,
    disturb_fore=0,
):
    encoder = sentencebert_encode

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    env.seed(seed)
    np.random.seed(seed)
    
    state, _ = env.reset(seed=seed)

    task_name = extract_task_name(state["instruction"])
    task = encoder([task_name]).to(device=device, dtype=torch.float32)
        
    state["observation"] = state["instruction"]

    state = parse_observation(state["observation"], task_name, 0, 0, 0)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    languages = (encoder([task_name])).to(device=device, dtype=torch.float32)

    languages = languages.reshape(1, -1).to(device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    
    subgoal_1, subgoal_2, subgoal_3 = 0, 0, 0
    
    for t in range(max_ep_len):
        if lang_mode == "val":
            idx = np.random.randint(int(200*(train_ratio)), int(200*(train_ratio + val_ratio)))
        else:
            idx = np.random.randint(int(200*(train_ratio + val_ratio)), 200)
        env.set_paraphrase_method(idx)
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

        action = np.array(action, dtype=float)

        action_str = form_action_str(action, task_name)

        actions[-1] = torch.tensor(action)
        action = list(action)

        state, reward, done, truncated, info = env.step(action_str)
        
        if diversity == "online_gpt":
            prompt = (
                PROMPT.replace("{feedback}", state["feedback"])
                .replace("{task_name}", task_name)
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
            
        elif diversity == "gpt_pool":
            feedback = generate_gpt_feedback(env, action_str, reward, info, env.last_infos, disturb_hind=disturb_hind, disturb_fore=disturb_fore)
            if informativeness == "no_lang":
                lang = ""
            elif informativeness == "f":
                lang = feedback.fp
            elif informativeness == "h":
                lang = feedback.hn
            else:
                lang = feedback.hn + " " + feedback.fp
        elif informativeness == "no_lang":
            lang = ""
        else:
            lang = state["feedback"]
        
        lang = encoder([lang])
        
        _state = parse_observation(state["observation"], task_name, subgoal_1, subgoal_2, subgoal_3)
        
        subgoal1, subgoal2, subgoal3 = get_subgoal_finished(task_name, state["observation"])
        
        subgoal_1 = max(subgoal_1, subgoal1)
        subgoal_2 = max(subgoal_2, subgoal2)
        subgoal_3 = max(subgoal_3, subgoal3)

        cur_state = torch.from_numpy(_state).to(device=device).reshape(1, state_dim)
        cur_lang = lang.to(device=device).reshape(1, 768)
        states = torch.cat([states, cur_state], dim=0)
        languages = torch.cat([languages, cur_lang], dim=0)

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        if done or truncated:
            break
        
    return episode_return, episode_length, True

