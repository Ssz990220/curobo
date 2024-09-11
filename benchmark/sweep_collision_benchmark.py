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

# Third Party
import numpy as np
import torch

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
    get_module_path
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_traj():
    module_path = get_module_path()
    # load bin file contains float from ../../data/traj.bin
    traj = np.fromfile(join_path(module_path, "../../data/traj_start_and_end.bin"), dtype=np.float32)
    traj = traj.reshape((4096 * 4,2,7))
    return traj


def load_curobo(robot_file, world_file, sweep_steps):
    # load curobo arm base?

    world_cfg = load_yaml(join_path(get_world_configs_path(), world_file))

    base_config_data = load_yaml(join_path(get_task_configs_path(), "base_cfg_sweep.yml"))
    graph_config_data = load_yaml(join_path(get_task_configs_path(), "graph_sweep.yml"))
    # base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0
    # if not compute_distance:
    #    base_config_data["constraint"]["primitive_collision_cfg"]["classify"] = False
    robot_config_data = load_yaml(join_path(get_robot_configs_path(), robot_file))

    base_config_data["constraint"]["primitive_collision_cfg"]["sweep_steps"] = sweep_steps

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


def bench_collision_curobo(robot_file, world_file, traj, b_size, use_cuda_graph=True, sweep_step = 1):
    arm_base = load_curobo(robot_file, world_file, sweep_step)
    arm_base.robot_self_collision_constraint.disable_cost()
    arm_base.bound_constraint.disable_cost()

    # random sample from traj as q_test
    random_indices = np.random.choice(traj.shape[0], size=b_size, replace=True)
    q_test = traj[random_indices,:,:]
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
            random_indices = np.random.choice(traj.shape[0], size=b_size, replace=True)
            q_test = traj[random_indices,:,:]
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

        for _ in range(10):
            q.copy_(q_warm.detach())
            g.replay()
            a = out.feasible
            # print(a)
            # a = ee_mat.clone()
        # q_new = torch.rand((b_size, robot_model.get_dof()), **vars(tensor_args))

        torch.cuda.synchronize()

        for _ in range(100):
            random_indices = np.random.choice(traj.shape[0], size=b_size, replace=True)
            q_test = traj[random_indices,:,:]
            q_test = tensor_args.to_device(q_test).unsqueeze(1)
            st_time = time.time()

            q.copy_(q_test.detach().requires_grad_(False))
            g.replay()
            a = out.feasible
            # print(a)
            # a = ee_mat.clone()
            torch.cuda.synchronize()
            dt = time.time() - st_time
            ts.append(dt)
        
    dt = np.median(ts)
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
        default="sweep_collision",
        help="File name prefix to use to save benchmark results",
    )

    args = parser.parse_args()

    traj = load_traj()
    traj_size = traj.size / 2 / 7

    # b_list = [4096]
    # b_list = [2]

    robot_list = get_robot_list()
    robot_list = [robot_list[0]]

    sweep_steps = [3, 7]
    b_list = [4096, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

    world_files = ["benchmark_shelf.yml", "benchmark_shelf_dense.yml", "benchmark_shelf_simple.yml"]

    print("running...")
    for world_file in world_files:
        print("Running: ", world_file)
        data = {"sweep_steps": [], "Collision Checking": [], "Batch Size": []}
        for robot_file in robot_list:
            print("Running: ", robot_file)
            for step in sweep_steps:
                print("Running: ", step)
                count = 0
                for b_size in b_list:
                    print("Running: ", robot_file, world_file, step, b_size)
                    dt_cu_cg = bench_collision_curobo(
                        robot_file,
                        world_file,
                        traj,
                        b_size,
                        use_cuda_graph=False,
                        sweep_step=step
                    )
                    count += 1
                    if count == 1:
                        continue
                # dt_kin_cg = bench_kin_curobo(
                #     robot_file, traj, traj_size, use_cuda_graph=True, use_coll_spheres=True
                # )

                    data["sweep_steps"].append(step * 2 + 2)
                    data["Collision Checking"].append(dt_cu_cg)
                    data["Batch Size"].append(b_size)
        write_yaml(data, join_path(args.save_path, "curobo_swept_" + world_file))
        try:
            # Third Party
            import pandas as pd

            df = pd.DataFrame(data)
            df.to_csv(join_path(args.save_path, args.file_name + ".csv"))
        except ImportError:
            pass
