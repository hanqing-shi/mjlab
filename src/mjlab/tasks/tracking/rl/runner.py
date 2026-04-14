import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    self.export_policy_to_onnx(policy_path, filename)
    
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )
    
    onnx_path = os.path.join(policy_path, filename)
    
    # 获取原始环境对象
    base_env = self.env.unwrapped
    metadata = get_base_metadata(base_env, run_name)

    # 通过 base_env 访问 observation_manager
    obs_manager = base_env.observation_manager
    actor_terms = obs_manager.active_terms["actor"]
    
    # 构造数组：例如 [3, 3, 3]
    # 1. 先得到整数列表
    history_lengths = [
      int(obs_manager.get_term_cfg("actor", name).history_length) 
      for name in actor_terms
    ]

    # 2. 手动转为逗号分隔的整数字符串，绕过 exporter_utils 的浮点数格式化
    metadata["history_length"] = ",".join(map(str, history_lengths))
   
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))