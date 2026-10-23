"""CPU regressions for the MOSS sampler, without a ComfyUI installation."""
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_moss():
    tree = ast.parse((ROOT / "moss.py").read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    namespace = {"SpectrumState": object}
    exec(compile(tree, "moss.py", "exec"), namespace)
    return types.SimpleNamespace(**namespace), namespace


@unittest.skipIf(torch is None, "torch not available")
class MossSamplingTest(unittest.TestCase):
    def setUp(self):
        self.moss, self.ns = load_moss()

    def test_native_sampling_uses_unit_timestep_multiplier(self):
        class AuraFlow:
            def patch(self, model, shift, multiplier=1000):
                model.multiplier = multiplier
                return (model,)

            def patch_aura(self, model, shift):
                return self.patch(model, shift, multiplier=1)

        model = types.SimpleNamespace(model=types.SimpleNamespace(
            diffusion_model=type("MOSSImage", (), {})()))
        self.ns["apply_moss_spectrum_patch"] = lambda model, **kw: model
        modules = {
            "nodes": types.SimpleNamespace(common_ksampler=lambda m, *a, **kw: m),
            "comfy_extras.nodes_slg": types.SimpleNamespace(SkipLayerGuidanceDiTSimple=object),
            "comfy_extras.nodes_model_advanced": types.SimpleNamespace(ModelSamplingAuraFlow=AuraFlow),
        }
        with patch.dict(sys.modules, modules):
            result = self.moss.MossSpectrumKSampler().sample(
                model, 0, 50, 3, "euler", "sgm_uniform", [], [], {},
                uncond_skip_layers="", compile=False)
        self.assertEqual(result.multiplier, 1)

    def test_native_fused_head_capture_and_cfg_batch_shape(self):
        dit = type("MOSSImage", (torch.nn.Module,), {} )()
        dit.layers = torch.nn.ModuleList([torch.nn.Identity()])
        dit.final_norm_video = torch.nn.LayerNorm(4)
        dit.time_embedding = dit.video_head_modulation = dit.final_linear_video = None
        dit.shared_block_modulation = None
        state = types.SimpleNamespace(
            patch_consumed=False, verbose=False, active=True, mode="actual",
            moss_feats=None, forecasters={}, m_param=1, lam=0.1, w=0.3,
            num_steps=50, history_size=10, step_idx=0,
            record_context=lambda *a: None)
        model = types.SimpleNamespace(model=types.SimpleNamespace(
            diffusion_model=dit, model_sampling=object()), model_options={})
        model.clone = lambda: model
        model.set_model_unet_function_wrapper = lambda fn: setattr(model, "wrapper", fn)
        self.ns.update({
            "MossSpectrumState": lambda **kw: state,
            "_resolve_live_components": lambda *a: (dit, model.model.model_sampling),
            "_maybe_rearm_for_new_execution": lambda s: None,
            "_advance_spectrum_state": lambda *a: False,
            "_normalize_cond_or_uncond": lambda *a: ([0, 1], True),
            "_spectrum_batch_keys": lambda *a: [0, 1],
            "_can_use_cached_prediction": lambda *a: False,
            "_mark_spectrum_patch_consumed": lambda s: None,
        })
        self.moss.apply_moss_spectrum_patch(model, compile=False)
        features = torch.arange(2 * 9 * 4).reshape(2, 9, 4).float()
        # Nine joint tokens: six image tokens followed by text/reference tokens.
        def actual(x, timestep, **kw):
            dit.layers[-1](features)
            return x - 1
        x = torch.zeros(2, 4, 2, 3)
        result = model.wrapper(actual, {"input": x, "timestep": torch.ones(2), "c": {}})
        torch.testing.assert_close(result, x - 1)
        for key in (0, 1):
            history = state.forecasters[key].h[0]
            self.assertEqual(history.shape, (1, 6, 4))
            torch.testing.assert_close(history[0], features[key, :6])
        features.zero_()
        self.assertNotEqual(state.forecasters[1].h[0].sum().item(), 0)

    def test_hf_capture_keeps_each_sample(self):
        dit = types.SimpleNamespace(final_norm_video=torch.nn.Identity())
        self.moss._ensure_capture_hook(dit)
        state = types.SimpleNamespace(moss_feats=[])
        dit.final_norm_video._spectrum_state = state
        features = torch.randn(3, 6, 4)
        dit.final_norm_video(features)
        torch.testing.assert_close(torch.stack(state.moss_feats), features)

    def test_native_cached_head_matches_direct_head(self):
        dit = type("MOSSImage", (torch.nn.Module,), {})()
        dit.time_embedding = torch.nn.Linear(256, 4)
        dit.video_head_modulation = torch.nn.Linear(4, 8)
        dit.final_norm_video = torch.nn.RMSNorm(4, eps=1e-6)
        dit.final_linear_video = torch.nn.Linear(4, 2, bias=False)
        feat = torch.randn(2, 6, 4)
        timestep = torch.tensor([0.8, 0.2])

        def embedding(t, dim):
            return t[:, None].expand(-1, dim)

        vec = dit.time_embedding(embedding(timestep, 256))
        shift, scale = dit.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        expected = dit.final_linear_video(dit.final_norm_video(feat) * (1 + scale) + shift)
        expected = expected.transpose(1, 2).reshape(2, 2, 2, 3)
        with patch.dict(sys.modules, {
            "comfy.ldm.flux.layers": types.SimpleNamespace(timestep_embedding=embedding)
        }):
            actual = self.moss._moss_head(dit, timestep, feat, 2, 3)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
