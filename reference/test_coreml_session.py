"""Portable failure/admission tests for the T18 development native-state wrapper."""
import unittest
import numpy as np
from coreml_qwen_export import CoreMLCheckSession

class FakeModel:
    def __init__(self):
        self.calls=0; self.failure=None
    def make_state(self): return [0]
    def predict(self,inputs,state):
        self.calls+=1; state[0]+=1
        if self.failure=='after-write': raise RuntimeError('partial native prediction')
        return {'logits':np.array([np.nan if self.failure=='nonfinite' else state[0]],dtype=np.float32)}

class Contracts(unittest.TestCase):
    def test_admission_preserves_state_and_does_not_predict(self):
        model=FakeModel(); session=CoreMLCheckSession(model,np); session.step(1)
        state=session.state
        for token,position in [(-1,None),(151936,None),(True,None),(1,2)]:
            with self.assertRaises(ValueError): session.step(token,position)
        self.assertIs(session.state,state); self.assertEqual(state,[1]); self.assertEqual(model.calls,1)
        self.assertEqual(session.processed,1); self.assertTrue(session.valid)
        session.processed=32
        with self.assertRaises(ValueError): session.step(1)
        self.assertEqual(state,[1]); self.assertEqual(model.calls,1)

    def test_partial_execution_invalidates_then_reset_recovers(self):
        for failure in ['after-write','nonfinite']:
            model=FakeModel(); session=CoreMLCheckSession(model,np); session.step(1)
            old=session.state; model.failure=failure
            with self.assertRaises((ValueError,RuntimeError)): session.step(2)
            self.assertEqual(old,[2]); self.assertFalse(session.valid); self.assertEqual(session.processed,0)
            with self.assertRaises(ValueError): session.step(3)
            self.assertEqual(model.calls,2)
            session.reset(); model.failure=None
            self.assertIsNot(session.state,old); self.assertEqual(session.step(3)['logits'],[1])
            self.assertTrue(session.valid); self.assertEqual(session.processed,1)

    def test_independent_states_share_only_the_model(self):
        model=FakeModel(); a,b=CoreMLCheckSession(model,np),CoreMLCheckSession(model,np)
        a.step(1); a.step(2); b.step(3)
        self.assertEqual(a.state,[2]); self.assertEqual(b.state,[1]); self.assertIsNot(a.state,b.state)
        b.reset(); self.assertEqual(a.state,[2]); self.assertEqual(b.state,[0])

if __name__=='__main__': unittest.main()
