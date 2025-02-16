#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
import argparse
import time
import os
import matplotlib.pyplot as plt

# Third Party
import numpy as np
import torch
import pandas as pd

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.rollout.arm_base import ArmBase, ArmBaseConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_robot_list,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_curobo(robot_file, world_file):
    # load curobo arm base?

    world_cfg = load_yaml(join_path(get_world_configs_path(), world_file))

    base_config_data = load_yaml(join_path(get_task_configs_path(), "base_cfg.yml"))
    graph_config_data = load_yaml(join_path(get_task_configs_path(), "graph.yml"))
    # base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0
    # if not compute_distance:
    #    base_config_data["constraint"]["primitive_collision_cfg"]["classify"] = False
    robot_config_data = load_yaml(join_path(get_robot_configs_path(), robot_file))

    arm_base = ArmBaseConfig.from_dict(
        robot_config_data["robot_cfg"],
        graph_config_data["model"],
        base_config_data["cost"],
        base_config_data["constraint"],
        base_config_data["convergence"],
        base_config_data["world_collision_checker_cfg"],
        world_cfg,
    )
    arm_base = ArmBase(arm_base)
    return arm_base


def bench_collision_curobo(robot_file, world_file, arm_sampler, b_size, use_cuda_graph=True):
    arm_base = load_curobo(robot_file, world_file)
    arm_base.robot_self_collision_constraint.disable_cost()
    arm_base.bound_constraint.disable_cost()
    
    q_test = arm_sampler.sample_random_actions(b_size).cpu().numpy()
    # load graph module:
    tensor_args = TensorDeviceType()
    q_test = tensor_args.to_device(q_test).unsqueeze(1)

    tensor_args = TensorDeviceType()
    
    q_warm = q_test + 0.5

    ts = []
    if not use_cuda_graph:
        for _ in range(10):
            out = arm_base.rollout_constraint(q_warm)
            torch.cuda.synchronize()

        torch.cuda.synchronize()

        for _ in range(100):
            q_test = arm_sampler.sample_random_actions(b_size).cpu().numpy()
            q_test = tensor_args.to_device(q_test).unsqueeze(1)
            st_time = time.time()
            out = arm_base.rollout_constraint(q_test)
            torch.cuda.synchronize()
            dt = time.time() - st_time
            ts.append(dt)
    else:
        q = q_warm.clone()

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                out = arm_base.rollout_constraint(q_warm)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            out = arm_base.rollout_constraint(q_warm)

        for _ in range(100):
            q_test = arm_sampler.sample_random_actions(b_size).cpu().numpy()
            q_test = tensor_args.to_device(q_test).unsqueeze(1)
            st_time = time.time()
            q.copy_(q_test.detach().requires_grad_(False))
            g.replay()
            a = out.feasible
            torch.cuda.synchronize()
            dt = time.time() - st_time
            ts.append(dt)
            # print(a)
            # a = ee_mat.clone()
        # q_new = torch.rand((b_size, robot_model.get_dof()), **vars(tensor_args))

    # return the median time
    dt = np.median(ts)

    # convert dt to float
    dt = float(dt)
    return dt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="kinematics",
        help="File name prefix to use to save benchmark results",
    )

    args = parser.parse_args()
    b_list = [4096, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

    robot_list = get_robot_list()
    robot_list = [robot_list[0]]

    world_files = ["benchmark_shelf_dense", "benchmark_shelf","benchmark_shelf_simple"];
    # world_files = ["benchmark_shelf_dense"];

    print("running...")
    for world in world_files:
        data = {"robot": [], "Collision Checking": [], "Batch Size": []}
        world_file = world + ".yml"
        for robot_file in robot_list:
            arm_sampler = load_curobo(robot_file, world_file)

            counter = 1
            # create a sampler with dof:
            for b_size in b_list:
                # sample test configs:

                dt_cu_cg = bench_collision_curobo(
                    robot_file,
                    world_file,
                    arm_sampler,
                    b_size,
                    use_cuda_graph=False,
                )
                # dt_kin_cg = bench_kin_curobo(
                #     robot_file, q_test, use_cuda_graph=True, use_coll_spheres=True
                # )

                if counter == 1:    # skip the first run
                    counter += 1
                    continue
                data["robot"].append(robot_file)
                data["Collision Checking"].append(dt_cu_cg)
                # data["Kinematics"].append(dt_kin_cg)
                data["Batch Size"].append(b_size)
        write_yaml(data, os.path.join(args.save_path, args.file_name + "_" + world + ".yml"))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(args.save_path, args.file_name + "_" + world + ".csv"))

        # plot the time per pose vs batch size
        plt.figure()
        batchSize = df["Batch Size"][1:-1]
        tBatch = df["Collision Checking"][1:-1]
        tPerPose = tBatch / batchSize * 1e6

        plt.plot(batchSize, tPerPose, label="Collision Checking")
        plt.xscale("log", base=2)
        plt.yscale("log", base=10)
        plt.xlabel("Batch Size (log2)")
        plt.ylabel("Time per Pose (us) (log10)")
        plt.title("Time per Pose vs Batch Size")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(args.save_path, args.file_name + "_" + world + "_time_per_pose.png"))
        plt.show()
