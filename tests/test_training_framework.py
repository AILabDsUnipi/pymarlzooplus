import time
import traceback
import unittest

from tests.config import algorithms
from pymarlzooplus import pymarlzooplus


def generate_training_configs(env_type, keys, common_args, algo_names, variants=None):
    if variants is None:
        variants = {"": {}}
    configs = {}
    for algo_name in algo_names:
        algo_conf = algorithms[algo_name]
        for key in keys:
            for suffix, override in variants.items():
                if suffix.endswith("_raw") and algo_name.endswith("_ns"):
                    continue
                if algo_name in  ["CDS", "COMA", "HAPPO", "IPPO", "EOI", "IA2C"] and suffix.endswith("_raw"):
                    override.update({"trainable_cnn": True, "centralized_image_encoding": False})

                test_name = f"{env_type}_{key}_{algo_name}"
                if suffix:
                    test_name += suffix

                config = {}
                config.update(algo_conf)
                config["env-config"] = env_type
                config["env_args"] = {**common_args, "key": key, **override}
                configs[test_name] = config
    return configs

class TestsTrainingFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_framework_params_dict = {}

        all_algos = list(algorithms.keys())
        common = {"time_limit": 2, "seed": 2024}
        pettingzoo_common = {
            "time_limit": 10,
            "render_mode": "human",
            "image_encoder": "ResNet18",
            "image_encoder_use_cuda": False,
            "image_encoder_batch_size": 2,
            "kwargs": "",
            "seed": 2024,
        }
        fully_coop_variants = {
            "_encoded": {"trainable_cnn": False},
            "_raw": {"trainable_cnn": True},
        }
        fc_configs = generate_training_configs(
            env_type="pettingzoo",
            keys=["pistonball_v6"],
            common_args=pettingzoo_common,
            variants=fully_coop_variants,
            algo_names=all_algos
        )
        cls.train_framework_params_dict.update(fc_configs)

        partial_obs_variants = {
            "_partial_observation_encoded": {"trainable_cnn": False, "partial_observation": True},
            "_partial_observation_raw": {"trainable_cnn": True, "partial_observation": True},
        }
        partial_configs = generate_training_configs(
            env_type="pettingzoo",
            keys=["entombed_cooperative_v3",],
            common_args=pettingzoo_common,
            variants=partial_obs_variants,
            algo_names=all_algos
        )
        cls.train_framework_params_dict.update(partial_configs)

        overcooked_variants = {"_sparse": {"reward_type": "sparse"}, "_shaped": {"reward_type": "shaped"}}
        overcooked_configs = generate_training_configs(
            env_type="overcooked",
            keys=["coordination_ring"],
            common_args=common,
            variants=overcooked_variants,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(overcooked_configs)

        pressureplate_configs = generate_training_configs(
            env_type="pressureplate",
            keys=["pressureplate-linear-4p-v0"],
            common_args=common,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(pressureplate_configs)


        lbf_v2_configs = generate_training_configs(
            env_type="gymma",
            keys=["lbforaging:Foraging-4s-11x11-3p-2f-coop-v2"],
            common_args=common,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(lbf_v2_configs)

        lbf_v3_configs = generate_training_configs(
            env_type="gymma",
            keys=["lbforaging:Foraging-4s-11x11-3p-2f-coop-v3"],
            common_args=common,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(lbf_v3_configs)


        rware_configs = generate_training_configs(
            env_type="gymma",
            keys=["rware:rware-small-4ag-hard-v1"],
            common_args=common,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(rware_configs)

        mpe_configs = generate_training_configs(
            env_type="gymma",
            keys=["mpe:SimpleSpread-3-v0"],
            common_args=common,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(mpe_configs)

        capturetarget_variants = {"": {},"_obs_one_hot": {"obs_one_hot": True}, "_wo_tgt_avoid_agent": {"tgt_avoid_agent": False}}
        capturetarget_configs = generate_training_configs(
            env_type="capturetarget",
            keys=["CaptureTarget-6x6-1t-2a-v0"],
            common_args=common,
            variants=capturetarget_variants,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(capturetarget_configs)

        boxpushing_configs = generate_training_configs(
            env_type="boxpushing",
            keys=["BoxPushing-6x6-2a-v0"],
            common_args=common,
            algo_names=all_algos,
        )
        cls.train_framework_params_dict.update(boxpushing_configs)

    def test_training_framework(self):
        completed = []
        failed = {}

        for name, params in self.train_framework_params_dict.items():
            print(
                "\n\n###########################################"
                "\n###########################################"
                f"\nRunning test for: {name}\n"
            )
            with self.subTest(environment=name):
                try:
                    pymarlzooplus(params)
                    completed.append(name)
                except (Exception, SystemExit) as e:
                    tb_str = traceback.format_exc()
                    failed[name] = tb_str
                    self.fail(f"Test for '{name}' failed with exception: {e}")
                time.sleep(5)
        if failed:
            msg = "Some tests failed:\n"
            for name, tb_str in failed.items():
                msg += f"\n=== {name} ===\n{tb_str}\n"
            self.fail(msg)

if __name__ == "__main__":
    unittest.main()
