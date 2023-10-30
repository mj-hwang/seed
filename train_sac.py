import os
import sys
import argparse
import gym
import numpy as np
import torch as th

sys.path.append(os.path.join(sys.path[0],'stable-baselines3'))
sys.path.append(os.path.join(sys.path[0],'robosuite'))

from stable_baselines3 import SAC

import robosuite as suite
from robosuite.environments.manipulation.POC_reaching import POCReaching
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


def main(args):
    if args.env == "Reaching2D":
        obs_keys = ["eef_xyz", "gripper_state"]
        total_timesteps = int(3e5)
    elif args.env == "StackCustom":
        obs_keys = ["cubeA_pos", "cubeB_pos"]
        total_timesteps = int(1e6)
    elif args.env == "Cleanup":
        obs_keys = ["eef_xyz", "eef_yaw", "gripper_state", "pnp_obj_pos_yaw", "push_obj_pos_yaw"]
        total_timesteps = int(4e6)

    env_train = suite.make(
        env_name=args.env,
        robots="Panda",
        reward_scale=5.0,
        use_delta=True,
        use_skills=False,
        use_yaw=False,
        normalized_params=True,
        normalized_obs=True,
    )
    env_train = GymWrapper(env_train, keys=obs_keys)

    env_eval = suite.make(
        env_name=args.env,
        robots="Panda",
        reward_scale=5.0, 
        use_delta=True,
        use_skills=False,
        use_yaw=False,
        normalized_params=True,
        normalized_obs=True,
    )
    env_eval = GymWrapper(env_eval, keys=obs_keys)

    # Save a checkpoint every 10000 steps
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=f"./logs/{args.env}/SAC/bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_gs_{args.gsteps}/seed_{args.seed}/",
        log_path=f"./logs/{args.env}/SAC/bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_gs_{args.gsteps}/seed_{args.seed}/",
        n_eval_episodes=10,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    callback = CallbackList([eval_callback])

    if args.save_checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=f"./logs/{args.env}/SAC/bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_gs_{args.gsteps}/seed_{args.seed}/",
            name_prefix="sac",
            save_vecnormalize=True,
        )
        callback.append(checkpoint_callback)

    model_sac = SAC(
        "MlpPolicy",
        env_train,
        learning_rate=args.lr,
        buffer_size=1000000,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        gradient_steps=args.gsteps,
        seed=args.seed,
        tensorboard_log=f"./logs/{args.env}/SAC/bs_{args.batch_size}_lr_{args.lr}_gamma_{args.gamma}_gs_{args.gsteps}/seed_{args.seed}/",
    )
    model_sac.learn(total_timesteps=total_timesteps, callback=callback, log_interval=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Reaching2D")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gsteps", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)


