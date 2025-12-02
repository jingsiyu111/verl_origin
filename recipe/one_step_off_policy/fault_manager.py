import os
import logging
import signal
import time
import re

import hydra
import ray
import threading
import torch
import subprocess

from functools import wraps
from verl.trainer.ppo.utils import Role


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FaultMgr:
    actor_fault = False
    rollout_fault = False

    @classmethod
    def bound(cls, trainer):
        cls.trainer = trainer

    @classmethod
    def update_retry_options(selfcls,ray_cls):
        old_init = ray_cls.__init__

        @wraps(old_init)
        def new_init(self, *args, **kwargs):
            additional_resource = {}
            old_init(self, *args, **kwargs)

            if cls.trainer.config:
                enable_retry = cls.trainer.config.fault_manager.enable_retry
                max_restarts = cls.trainer.config.fault_manager.max_restarts
                max_task_retries = cls.trainer.config.fault_manager.max_task_retries
                if enable_retry:
                    additional_resource = {
                            "max_restarts": max_restarts,
                            "max_task_retries": max_task_retries,
                        }

            self.update_options(additional_resource)

        ray_cls.__init__ = new_init

    @classmethod
        def fault_execute_remote_single_worker(selfcls, ray_cls):
            old_func = ray_cls._execute_remote_single_worker

            @wraps(old_func)
            def new_execute(self, worker, method_name, *args, **kwargs):
                if self.fused_worker_used and method_name not in self.method_names:
                    remote_call = getattr(worker, self.fused_worker_execute_fn_name)
                    return remote_call.remote(f"{self.sub_cls_name}_fwmn_{method_name}", *args, **kwargs)
                # fused worker not used
                remote_call = getattr(worker, method_name)
                return remote_call.options(retry_exceptions=True).remote(*args, **kwargs)

            ray_cls._execute_remote_single_worker = new_execute

    @classmethod
    def rebuild_resourse_pool(cls, role):
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        def release_placement_groups(resource_pool):
            print("[info]release placement group...")
            # "Release all placement groups in the resource pool"
            if resource_pool.pgs is None:
                print("No placement groups to release")
                return

            for pg in resource_pool.pgs:
                try:
                    ray.util.remove_placement_group(pg)
                except Exception as e:
                    print(f"Error releasing placement group {pg}: {e}")

            resource_pool.pgs = None
            print("Placement groups released")

        print("[info]rebuild_resourse_pool ...")

        # breakpoint()
        resource_pool = cls.trainer.resource_pool_manager.get_resource_pool(role)
        release_placement_groups(resource_pool)

        # create new actor/ref RayClassWithInitArgs
        if role == Role.Actor:
            del cls.trainer.actor_wg
            del cls.trainer.ref_policy_wg
            # get resource pool
            resource_pool = cls.trainer.resource_pool_manager.get_resource_pool(Role.Actor)

            actor_cls = RayClassWithInitArgs(
                cls=cls.trainer.role_worker_mapping[Role.Actor],
                config=cls.trainer.config.actor_rollout_ref,
                role=str(Role.Actor)
            )

            ref_cls = RayClassWithInitArgs(
                cls=cls.trainer.role_worker_mapping[Role.RefPolicy],
                config=cls.trainer.config.actor_rollout_ref,
                role=str(Role.RefPolicy)
            )

            class_dict = {
                str(Role.Actor): actor_cls,
                str(Role.RefPolicy): ref_cls,
            }

        # create new rollout RayClassWithInitArgs
        if role == Role.Rollout:
            del cls.trainer.rollout_wg
            print("[info]rollout resource pool delete ...")

            # get resource pool
            resource_pool = cls.trainer.resource_pool_manager.get_resource_pool(Role.Rollout)
            rollout_cls = RayClassWithInitArgs(
                cls=cls.trainer.role_worker_mapping[Role.Rollout],
                config=cls.trainer.config.actor_rollout_ref,
                role=str(Role.Rollout)
            )

            class_dict = {
                str(Role.Rollout): rollout_cls,
            }

        # update resource pool to cls dict
        if resource_pool not in cls.trainer.resource_pool_to_cls:
            cls.trainer.resource_pool_to_cls[resource_pool] = {}
        cls.trainer.resource_pool_to_cls[resource_pool].update(class_dict)

        # create worker dict cls and wg
        worker_dict_cls = create_colocated_worker_cls(class_dict)
        wg_dict = cls.trainer.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name=cls.trainer.device_name,
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())

        if role == Role.Actor:
            cls.trainer.actor_wg = spawn_wg[str(Role.Actor)]
            cls.trainer.ref_policy_wg = spawn_wg[str(Role.RefPolicy)]
        elif role == Role.Rollout:
            cls.trainer.rollout_wg = spawn_wg[str(Role.Rollout)]

    @classmethod
    def sync_weight(cls, role):
        def create_weight_sync_group():
            master_address = ray.get(cls.trainer.actor_wg.workers[0]._get_node_ip.remote())
            master_port = ray.get(cls.trainer.actor_wg.workers[0]._get_free_port.remote())
            world_size = len(cls.trainer.actor_wg.workers + cls.trainer.rollout_wg.workers)
            cls.trainer.actor_wg.create_weight_sync_group(
                master_address,
                master_port,
                0,
                world_size,
            )
            ray.get(
                cls.trainer.rollout_wg.create_weight_sync_group(
                    master_address,
                    master_port,
                    len(cls.trainer.actor_wg.workers),
                    world_size,
                )
            )

        if role == Role.Actor:
            cls.trainer.actor_wg.init_model()
            cls.trainer.ref_policy_wg.init_model()
            weights_info = cls.trainer.rollout_wg.get_actor_weights_info()[0]
            cls.trainer.actor_wg.set_actor_weights_info(weights_info)
            create_weight_sync_group()

        if role == Role.Rollout:
            cls.trainer.rollout_wg.init_model()
            weights_info = cls.trainer.actor_wg.get_actor_weights_info()[0]
            cls.trainer.rollout_wg.set_actor_weights_info(weights_info)
            create_weight_sync_group()

    @classmethod
    def rebuild_worker_group(cls, role):
        # breakpoint()
        if role == Role.Actor:
            cls.rebuild_resourse_pool(role)
            cls.sync_weight(role)
            print("[INFO] Actor and reference policy worker groups recovered")

        elif role == Role.Rollout:
            cls.rebuild_resourse_pool(role)
            cls.sync_weight(role)
            print("[INFO] rollout worker groups recovered")

    @classmethod
    def catch_actor_fault(cls, batch):
        actor_max_rebuild_times = cls.trainer.config.fault_manager.actor_max_rebuild_times
        enable_retry = cls.trainer.config.fault_manager.enable_retry
        try:
            old_log_prob = cls.trainer.actor_wg.compute_log_prob(batch)
            return old_log_prob
        except Exception as e:
            if enable_retry:
                if cls.rollout_fault:
                    raise e
                for attempt in range(actor_max_rebuild_times):
                    try:
                        print(f'attempt {attempt} calling actor_wg.compute_log_prob')
                        start_time = time.time()
                        cls.rebuild_worker_group(Role.Actor)
                        end_time = time.time()
                        print(
                            f"[INFO] actor_wg rebuild task attempt {attempt} retried successfully in {end_time - start_time:.2f}s.")

                        old_log_prob = cls.trainer.actor_wg.compute_log_prob(batch)
                        return old_log_prob

                    except Exception as e:
                        if attempt == actor_max_rebuild_times - 1:
                            cls.actor_fault = True
                            raise RuntimeError(
                                f"Failed to recover actor worker after {actor_max_rebuild_times} attempts.") from e
            else:
                raise e

    @classmethod
    def catch_rollout_fault(cls, batch_data_future, wait_flag=False):
        rollout_max_rebuild_times = cls.trainer.config.fault_manager.rollout_max_rebuild_times
        enable_retry = cls.trainer.config.fault_manager.enable_retry
        if wait_flag:
            ready_refs, _ = ray.wait(batch_data_future.gen_batch_output.futures)
            if not ready_refs:
                return
            else:
                try:
                    ray.get(ready_refs[0])
                except Exception as e:
                    if enable_retry:
                        if cls.actor_fault:
                            raise e
                        for attempt in range(rollout_max_rebuild_times):
                            try:
                                print(f'attempt {attempt} calling rollout_wg.batch_data_future, wait_flag=True')

                                start_time = time.time()
                                cls.rebuild_worker_group(Role.Rollout)
                                end_time = time.time()
                                print(
                                    f"[INFO] rollout_wg rebuild task attempt {attempt} retried successfully in {end_time - start_time:.2f}s.")

                                batch_data_future.gen_batch_output = cls.trainer.rollout_wg.async_generate_sequences(
                                    cls.trainer.gen_batch)
                                end_time_2 = time.time()
                                print(
                                    f"[INFO] batch_data_future attempt {attempt} retried successfully in {end_time_2 - end_time:.2f}s, wait_flag=True.")
                                batch_data_future.future_reward = None
                                if cls.trainer.config.reward_model.launch_reward_fn_async:
                                    # Store the object reference and set up callback
                                    batch_data_future.future_reward = cls.trainer._launch_individual_rewards.remote(
                                        batch_data_future.gen_batch_output, cls.trainer.config, cls.trainer.tokenizer,
                                        cls.trainer.non_tensor_batch
                                    )
                                break
                            except Exception as e:
                                if attempt == rollout_max_rebuild_times - 1:
                                    cls.rollout_fault = True
                                    raise RuntimeError(
                                        f"Failed to recover rollout worker after {rollout_max_rebuild_times} attempts.") from e
                    else:
                        raise e

        else:
            try:
                batch_data_future.get()
                return
            except Exception as e:
                if enable_retry:
                    if cls.actor_fault:
                        raise e

                    print("[info]rollout fault ...")
                    for attempt in range(rollout_max_rebuild_times):
                        try:
                            print(f'attempt {attempt} calling rollout_wg.batch_data_future, wait_flag=False')
                            start_time = time.time()
                            cls.rebuild_worker_group(Role.Rollout)
                            end_time = time.time()

                            print(
                                f"[INFO] rollout_wg rebuild task attempt {attempt} retried successfully in {end_time - start_time:.2f}s.")

                            batch_data_future.gen_batch_output = cls.trainer.rollout_wg.async_generate_sequences(
                                cls.trainer.gen_batch)
                            batch_data_future.future_reward = None
                            if cls.trainer.config.reward_model.launch_reward_fn_async:
                                # Store the object reference and set up callback
                                batch_data_future.future_reward = cls.trainer._launch_individual_rewards.remote(
                                    batch_data_future.gen_batch_output, cls.trainer.config, cls.trainer.tokenizer,
                                    cls.trainer.non_tensor_batch
                                )
                            break

                        except Exception as e:
                            if attempt == rollout_max_rebuild_times - 1:
                                cls.rollout_fault = True
                                raise RuntimeError(
                                    f"Failed to recover rollout worker after {rollout_max_rebuild_times + 1} attempts, wait_flag=False.") from e
                else:
                    raise e

    @classmethod
    def catch_reward_fault(cls, batch, future_reward):
        from verl.trainer.ppo.reward import compute_reward

        reward_max_rebuild_times = cls.trainer.config.fault_manager.max_task_retries
        enable_retry = cls.trainer.config.fault_manager.enable_retry

        def reward_origin(batch, future_reward):

            if cls.trainer.use_rm:
                reward_tensor = cls.trainer.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if cls.trainer.config.reward_model.launch_reward_fn_async:
                # future_reward was already started in _async_gen_next_batch
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, cls.trainer.reward_fn)

            return reward_tensor, reward_extra_infos_dict, batch

        try:
            return reward_origin(batch, future_reward)

        except Exception as e:
            if enable_retry:
                if cls.rollout_fault:
                    raise e
                for attempt in range(reward_max_rebuild_times):
                    try:
                        print(f"[INFO] reward task attempt {attempt} retried. ")
                        time.sleep(15)
                        return reward_origin(batch, future_reward)
                    except Exception as e:
                        if attempt == reward_max_rebuild_times - 1:
                            raise RuntimeError(
                                f"Failed to recover reward compute after {reward_max_rebuild_times} attempts.") from e
            else:
                raise e

    @classmethod
    def max_reschedule(cls):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                config = args[0] if args else kwargs.get("config", None)
                max_reschedule_times = config.fault_manager.max_reschedule_times
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    if config.fault_manager.enable_retry:
                        for attempt in range(max_reschedule_times):
                            print(f"attempt {attempt} calling main_ppo")
                            try:
                                ray.shutdown()
                                cls.rollout_fault = False
                                cls.actor_fault = False
                                func(*args, **kwargs)
                                break
                            except Exception as e:
                                if attempt == max_reschedule_times - 1:
                                    raise RuntimeError(
                                        f"Failed to recover rollout worker after {max_reschedule_times} attempts.") from e
                    else:
                        raise e

            return wrapper

        return decorator

    @classmethod
    def timeout(cls, seconds=1, error_message="timeout!"):
        def decorator(func):
            ai_core_flag = {"flag": False}

            def _handle_timeout(signum, frame):
                if ai_core_flag['flag'] == True:
                    raise TimeoutError(error_message)
                else:
                    signal.alarm(seconds)

            def sys_command(cmd):
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    return result.stdout
                except Exception as e:
                    raise Exception(f"can't find AICore usage, {e}")

            @wraps(func)
            def wrapper(self, *args, **kwargs):
                npu_flag = False
                if hasattr(torch, "npu") and torch,npu.is_available():
                    npu_flag = True
                if self.config.fault_manager.check_aicore and npu_flag:
                    stop_flag = threading.Event()

                    def monitor():
                        start_time = time.time()
                        while not stop_flag.is_set():
                            pid = os.getpid()
                            chip_find_cmd = f"npu-smi info | grep '{pid}'"
                            text = sys_command(chip_find_cmd)
                            match = re.search(r'\|\s*(\d+)\s+(\d+)\s*\|', text)
                            if match:
                                device_id, chip_id = match.groups()
                                aicore_cmd = f"npu-smi info -i {device_id} -c {chip_id} -t usages | grep 'Aicore'"
                                res = sys_command(aicore_cmd).split(':')[1]
                                usage = int(res.strip())
                                if usage == 0:
                                    if not start_time:
                                        start_time = time.time()
                                    elif time.time() - start_time > seconds:
                                        ai_core_flag['flag'] = True
                                        break
                                else:
                                    start_time = None
                            else:
                                continue
                            time.sleep(1)

                    t = threading.Thread(target=monitor)
                    t.start()

                    signal.signal(signal.SIGALRM, _handle_timeout)
                    signal.alarm(seconds)

                    try:
                        return func(self, *args, **kwargs)
                    finally:
                        stop_flag.set()
                        t.join()
                else:
                    return func(self, *args, **kwargs)
            return wrapper

        return decorator