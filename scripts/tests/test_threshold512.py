"""Synthetic tests; no real biometric training is performed by this suite."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from Thresholding_Model_512 import ThresholdingModel512, Thresholding512Config, MODES, MODALITIES
from scripts.s05_threshold512 import train as ex
from scripts.tests.helpers import synthetic_data


class Threshold512Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = ex.parse_args(['--device','cpu','--seeds','17','--threshold-epochs','1',
                                  '--reliability-epochs','1','--patience','1','--batch-size','4',
                                  '--log-interval','100','--output-dir',str(Path(self.tmp.name)/'run')])
        self.args.temperature, self.args.data_seed = .02, 17
        self.data, self.users, self.pairs = synthetic_data()
        self.cache = {m:torch.from_numpy(x.copy()) for m,x in self.data.arrays.items()}

    def model(self, mode):
        return ex.make_model(self.args, mode, 17)

    def inputs(self):
        return self.cache['face'][:4], self.cache['fingerprint'][:4]

    def test_shapes_and_identical_initialization(self):
        previous = None
        for mode in MODES:
            model = self.model(mode)
            out = model(*self.inputs())
            for m, x in zip(MODALITIES,self.inputs()):
                for name in ('tau','bits','soft_bits','margin','reliability'):
                    self.assertEqual(out[f'{m}_{name}'].shape,(4,512))
                self.assertTrue(torch.equal(out[f'{m}_bits'],(x>=0).to(torch.uint8)))
            if previous is not None:
                for key, value in model.reliability.state_dict().items():
                    torch.testing.assert_close(value, previous[key],atol=0,rtol=0)
            previous = model.reliability.state_dict()

    def test_fixed_thresholds_vary_by_coordinate_but_never_observation(self):
        model = self.model('fixed')
        with torch.no_grad():
            for m in MODALITIES:
                model.thresholds[m].base_logit.copy_(torch.linspace(-1,1,512))
        f,p = self.inputs()
        out, changed = model(f,p), model(-f,-p)
        for m in MODALITIES:
            tau = out[f'{m}_tau']
            self.assertEqual(len(torch.unique(tau[0])),512)
            torch.testing.assert_close(tau,changed[f'{m}_tau'],atol=0,rtol=0)
            torch.testing.assert_close(tau,tau[:1].expand_as(tau),atol=0,rtol=0)
            self.assertLess(float(tau.abs().max()),model.config.tau_bound)

    def test_dynamic_thresholds_are_not_broadcast_scalar_and_branches_are_independent(self):
        model = self.model('dynamic')
        with torch.no_grad():
            for m in MODALITIES:
                torch.nn.init.normal_(model.thresholds[m].delta.weight,std=.2)
        f,p = self.inputs()
        out, changed_f, changed_p = model(f,p), model(-f,p), model(f,-p)
        for suffix in ('tau','bits','reliability'):
            torch.testing.assert_close(out[f'face_{suffix}'],changed_p[f'face_{suffix}'],atol=0,rtol=0)
            torch.testing.assert_close(out[f'fingerprint_{suffix}'],changed_f[f'fingerprint_{suffix}'],atol=0,rtol=0)
        for m, changed in (('face',changed_f),('fingerprint',changed_p)):
            self.assertGreater(float(out[f'{m}_tau'].std(1).mean()),0)
            self.assertFalse(torch.equal(out[f'{m}_tau'],changed[f'{m}_tau']))
            self.assertLess(float(out[f'{m}_tau'].abs().max()),model.config.tau_bound)

    def test_threshold_loss_backprop_updates_coordinate_parameters(self):
        rows = self.pairs['train'][[0,4,8]]
        for mode in ('fixed','dynamic'):
            model = self.model(mode).set_stage('threshold')
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                e,q = ex.common.forward_pair(model,self.cache,rows)
                ex.threshold_loss(model,e,q,rows,self.users,self.args)['total'].backward()
                for m in MODALITIES:
                    grad = model.thresholds[m].base_logit.grad
                    self.assertEqual(grad.shape,(512,))
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(int(torch.count_nonzero(grad)),100)
                    if mode=='dynamic':
                        self.assertGreater(float(model.thresholds[m].delta.weight.grad.abs().sum()),0)
                self.assertTrue(all(p.grad is None for p in model.reliability.parameters()))
                optimizer.step()
            out=model(*self.inputs())
            self.assertGreater(float(out['face_tau'].std(1).mean()),0)

    def test_reliability_cannot_change_thresholds(self):
        model=self.model('dynamic')
        stats=ex.train_ops.fit_statistics(model,self.cache,self.data,self.users,self.pairs['train'],self.args)
        for m in MODALITIES:
            model.reliability[m].set_statistics(stats[m])
        model.set_stage('reliability')
        frozen=model.fixed_reliability_state()
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
        e,q=ex.common.forward_pair(model,self.cache,self.pairs['train'][:4])
        ex.loss_ops.reliability_loss(e,q).backward()
        optimizer.step()
        ex.train_ops.assert_fixed(model,frozen)
        # Even a caller who accidentally unfreezes tau cannot backprop BCE to it.
        for p in model.parameters():
            p.grad=None
            p.requires_grad_(True)
        e,q=ex.common.forward_pair(model,self.cache,self.pairs['train'][:4])
        ex.loss_ops.reliability_loss(e,q).backward()
        self.assertTrue(all(p.grad is None for p in model.thresholds.parameters()))

    def test_invalid_inputs_and_checkpoint_round_trip(self):
        f,p=self.inputs()
        for mode in MODES:
            model=self.model(mode)
            for bad in (f[:,:511],f.double(),torch.full_like(f,float('nan'))):
                with self.assertRaises(ValueError):
                    model(bad,p)
            path=Path(self.tmp.name)/f'{mode}.pt'
            model.save_checkpoint(path,{'note':'synthetic'})
            restored,meta=ThresholdingModel512.load_checkpoint(path)
            self.assertEqual(meta,{'note':'synthetic'})
            before,after=model(f,p),restored(f,p)
            for key in before:
                torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
        with self.assertRaises(ValueError):
            Thresholding512Config(mode='scalar')
        wrong=Path(self.tmp.name)/'wrong.pt'
        torch.save({'format':'multimodal-threshold-reliability-abc-v1'},wrong)
        with self.assertRaises(ValueError):
            ThresholdingModel512.load_checkpoint(wrong)

    def test_collapse_guard_and_tau_statistics(self):
        rows=self.pairs['validation']
        plan=ex.metric_ops.evaluation_plan(rows,self.users,'cpu')
        model=self.model('zero')
        e,q=ex.train_ops.collect(model,self.cache,rows,4)
        baseline=ex.bit_metrics(e,q,plan,self.args)
        self.assertTrue(ex.selection_guard(baseline,baseline,self.args)[0])
        candidate=json.loads(json.dumps(baseline))
        candidate['bits']['face']['constant_coordinate_fraction']=1
        allowed,reasons=ex.selection_guard(candidate,baseline,self.args)
        self.assertFalse(allowed)
        self.assertTrue(any('constant_coordinate_fraction' in r for r in reasons))
        for m in MODALITIES:
            self.assertEqual(baseline['tau'][m]['query']['within_vector_std_mean'],0)

    def test_cli_rejects_unsafe_output_and_invalid_settings(self):
        for flags in (['--output-dir',self.tmp.name],['--tau-base-bound','nan'],
                      ['--seeds','17','17'],['--hidden-dim','0'],['--variance-shrinkage','2']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                ex.parse_args(flags)

    def test_synthetic_complete_pipeline_and_saved_512_vectors(self):
        with redirect_stdout(io.StringIO()):
            report=ex.run(self.args,self.data,self.users,self.pairs,{'temperature':.02,'seed':17},{},synthetic=True)
        self.assertEqual(len(report['trials']),3)
        self.assertTrue(report['input_and_source_unchanged'])
        self.assertTrue(report['synthetic_test_only'])
        status=json.loads((self.args.output_dir/'status.json').read_text())
        self.assertEqual(status['state'],'completed')
        for trial in report['trials']:
            folder=Path(trial['checkpoint']).parent
            model,meta=ThresholdingModel512.load_checkpoint(trial['checkpoint'])
            self.assertEqual(model.config.mode,trial['arm'])
            self.assertEqual(bool(model.reliability_fitted),trial['reliability_training']['best_epoch']>0)
            with np.load(folder/'test_outputs.npz') as z:
                for m in MODALITIES:
                    self.assertEqual(z[f'query_{m}_tau'].shape,(len(self.pairs['test']),512))
            self.assertTrue((folder/'test_tau_coordinates.json').is_file())
        self.assertTrue((self.args.output_dir/'paired_deltas.csv').is_file())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_both_backward_stages(self):
        self.args.device='cuda'
        cache={m:x.cuda() for m,x in self.cache.items()}
        rows=self.pairs['train'][[0,4,8]]
        for stage in ('threshold','reliability'):
            model=self.model('dynamic').set_stage(stage)
            e,q=ex.common.forward_pair(model,cache,rows)
            loss=(ex.threshold_loss(model,e,q,rows,self.users,self.args)['total'] if stage=='threshold'
                  else ex.loss_ops.reliability_loss(e,q))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':
    unittest.main()
