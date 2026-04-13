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
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)

    actor_terms = self.env.observation_manager.active_terms["actor"]
    term_cfg = self.env.observation_manager.get_term_cfg("actor", actor_terms[0])

    metadata.update(
        {
          "history_length": term_cfg.history_length,
          "flatten_history": int(term_cfg.flatten_history_dim)
        }
      )
    # metadata["history_length"] = term_cfg.history_length
    # metadata["flatten_history"] = int(term_cfg.flatten_history_dim)
   
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
