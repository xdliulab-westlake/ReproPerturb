import unittest
import numpy as np
from composition_metrics import validate, align, metrics, clr_additive, simplex_projection, rank_metrics


class CompositionContract(unittest.TestCase):
    def test_identity_extreme_and_shared_reference(self):
        p=np.array([[.2,.8],[.8,.2],[.5,.5]])
        self.assertEqual(metrics(p,p)['mae_pp'],0)
        self.assertEqual(metrics(p,p)['jsd_bits'],0)
        self.assertAlmostEqual(metrics(np.array([[1.,0.]]),np.array([[0.,1.]]))['jsd_bits'],1)
        r=clr_additive(np.array([.5,.5]),np.array([[.7,.3],[.2,.8]]))
        validate(r[None]);self.assertTrue(np.allclose((p-r)-(p-r),0))
        q=np.roll(p,1,axis=0)
        self.assertAlmostEqual(np.abs((p-r)-(q-r)).mean(),np.abs(p-q).mean())

    def test_key_alignment_and_invalid_outputs(self):
        p=np.array([[.2,.8],[.7,.3]])
        got=align(['b','a'],['y','x'],p,['a','b'],['x','y'])
        np.testing.assert_allclose(got,[[.3,.7],[.8,.2]])
        for bad in [np.array([[np.nan,0]]),np.array([[-.1,1.1]]),np.array([[.2,.3]])]:
            with self.assertRaises(ValueError):validate(bad)
        with self.assertRaises(ValueError):align(['a','a'],['x','y'],p,['a','b'],['x','y'])

    def test_projection_and_ties(self):
        np.testing.assert_allclose(simplex_projection(np.array([-.2,.4,.8])),[0,.3,.7])
        truth=np.array([.1,.2,.3,.4])
        r=rank_metrics(np.ones(4),truth,k=3)
        self.assertAlmostEqual(r['yield_pp'],25)
        self.assertAlmostEqual(r['regret_pp'],5)
        self.assertIsNone(rank_metrics(np.ones(4),np.ones(4),k=3)['ndcg'])
        self.assertEqual(rank_metrics(truth,truth,k=3)['regret_pp'],0)


if __name__=='__main__':unittest.main()
