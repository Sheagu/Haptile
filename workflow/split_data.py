import argparse
import glob
import os
import random


def is_valid_trajectory_dir(root, path):
    full_path = os.path.join(root, path)
    if not os.path.isdir(full_path):
        return False
    if (
        path.endswith("failed")
        or path.endswith("ood")
        or path.endswith("ikbad")
        or path.endswith("heated")
        or path.endswith("stop")
        or path.endswith("hard")
    ):
        return False
    return os.path.isfile(os.path.join(full_path, "trajectory.h5"))


def get_train_test_paths(all_paths, seed):
    assert len(all_paths) >= 2
    all_paths = all_paths.copy()
    random.Random(seed).shuffle(all_paths)
    train_paths = all_paths[:-1]
    test_paths = all_paths[-1:]
    print(
        f"using {len(train_paths)} trajectories for train and "
        f"{len(test_paths)} trajectories for test with seed {seed}"
    )
    return train_paths, test_paths


def split_symlink_train_test_merge_dataset(root_list, t_root, num_traj_per_task, seed):
    target_train_root = t_root + f"_train"
    target_test_root = t_root + f"_test"
    print(
        f"split data points from",
        root_list,
        f"to {target_train_root}: train and {target_test_root}: test",
    )
    if os.path.exists(target_train_root):
        return
    assert not os.path.exists(target_test_root)

    os.makedirs(target_train_root)
    os.makedirs(target_test_root)

    all_paths = []
    for root in sorted(root_list):
        for path in sorted(os.listdir(root))[:num_traj_per_task]:
            if is_valid_trajectory_dir(root, path):
                all_paths.append((root, path))

    train_paths, test_paths = get_train_test_paths(all_paths, seed)

    for root, sl_path in train_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_train_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)

    for root, sl_path in test_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_test_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)

    print()
    print("Done!!")


def split_symlink_train_test_dataset(root, t_root, seed):
    target_train_root = t_root + f"_train"
    target_test_root = t_root + f"_test"
    print(
        f"split data points from {root} to {target_train_root}: train and {target_test_root}: test"
    )

    if os.path.exists(target_train_root):
        return
    assert not os.path.exists(target_test_root)

    os.makedirs(target_train_root)
    os.makedirs(target_test_root)

    all_paths = []
    for path in sorted(os.listdir(root)):
        if is_valid_trajectory_dir(root, path):
            all_paths.append(path)

    train_paths, test_paths = get_train_test_paths(all_paths, seed)

    for sl_path in train_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_train_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)
    for sl_path in test_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_test_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)
    print()
    print("Done!!")


def split_symlink_dataset(root, num_trajs, seed):
    target_root = root + f"_{num_trajs}"
    print(f"split {num_trajs} data points from {root} to {target_root}")

    if os.path.exists(target_root):
        return
    assert not os.path.exists(target_root)

    os.makedirs(target_root)

    all_paths = []
    for path in os.listdir(root):
        if is_valid_trajectory_dir(root, path):
            all_paths.append(path)

    random.Random(seed).shuffle(all_paths)

    assert len(all_paths) >= num_trajs, (
        f"requested {num_trajs} trajectories from {root}, "
        f"but only found {len(all_paths)}"
    )
    sl_paths = all_paths[:num_trajs]
    for sl_path in sl_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)
    print()
    print("Done!!")


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--base_path", type=str, default="/hato/")
    arg.add_argument("--output_path", type=str, default="/split_data")
    arg.add_argument(
        "--data_name",
        nargs="+",
        type=str,
        default=[
            "data_banana",
        ],
    )
    arg.add_argument("--num_trajs", nargs="*", type=int, default=[])
    arg.add_argument("--merge", action="store_true")
    arg.add_argument("--merge_name", type=str, default="data_banana_all")
    arg.add_argument("--num_traj_per_task", type=int, default=20)
    arg.add_argument("--seed", type=int, default=0)
    args = arg.parse_args()

    if not args.merge:
        for data_name in args.data_name:
            split_symlink_train_test_dataset(
                os.path.join(args.base_path, data_name),
                os.path.join(args.output_path, data_name),
                args.seed,
            )
            for num_trajs in args.num_trajs:
                split_symlink_dataset(
                    os.path.join(args.output_path, data_name) + "_train",
                    num_trajs,
                    args.seed,
                )

    else:
        data_name_list = [
            os.path.join(args.base_path, data_name) for data_name in args.data_name
        ]
        split_symlink_train_test_merge_dataset(
            data_name_list,
            os.path.join(args.output_path, args.merge_name),
            args.num_traj_per_task,
            args.seed,
        )
