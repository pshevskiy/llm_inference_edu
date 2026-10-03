"""Portable T19 request contracts; no model, conversion or Apple hardware."""
import unittest
import numpy as np
from tokenizers import Tokenizer, AddedToken, models, decoders
from coreml_benchmark import sample, TextStream, validate_workload


class Session:
    def reset(self):
        self.processed = 0
        self.tokens = []

    def step(self, token):
        self.tokens.append(token)
        self.processed += 1
        # Independent tie case: lowest vocabulary index must win.
        return {'logits': np.array([0., 3., 3.], dtype=np.float32)}


class Text:
    def get_added_tokens_decoder(self): return {}
    def id_to_token(self, token): return str(token)
    def decode(self, ids, skip_special_tokens): return ''.join(str(i) for i in ids)


class Contracts(unittest.TestCase):
    def workload(self):
        return {'name': 'hand trace', 'prompt': [2, 0], 'fixed_output': [2, 0, 2],
            'continuation': [0, 2], 'capacity': 32, 'warmups': 0, 'repetitions': 1}

    def test_pending_token_is_evaluated_before_continuation(self):
        for mode, emitted, evaluated in [
            ('TeacherForced', [2, 0, 2], [2, 0, 2, 0, 2, 0, 2]),
            ('Greedy', [1, 1, 1], [2, 0, 1, 1, 1, 0, 2]),
        ]:
            session = Session()
            result = sample(session, self.workload(), mode, Text(), np)
            self.assertEqual(result['emitted'], emitted)
            self.assertEqual(session.tokens, evaluated)
            self.assertEqual(result['processed_before_continuation'], 4)
            self.assertEqual(result['processed_after_continuation'], 7)
            self.assertEqual(result['continuation_emitted_token'], 1)
            self.assertEqual(result['pending_emitted_tokens'], 1)
            self.assertEqual(len(result['backend_decode_ns']), 2)
            self.assertEqual(result['ttft_ns'] is None, mode == 'TeacherForced')

    def test_eos_is_emitted_but_pending_then_continued(self):
        class StopSession(Session):
            def step(self, token):
                super().step(token)
                row = np.zeros(151646, dtype=np.float32)
                row[151645] = 1
                return {'logits': row}
        session = StopSession()
        result = sample(session, self.workload(), 'Greedy', Text(), np)
        self.assertEqual(result['emitted'], [151645])
        self.assertTrue(result['stopped_by_eos'])
        self.assertEqual(session.tokens, [2, 0, 151645, 0, 2])
        self.assertEqual(result['backend_decode_ns'], [])
        self.assertIsNone(result['decode_tokens_per_second'])

    def test_utf8_bytes_remain_pending_and_special_tokens_hidden(self):
        tokenizer = Tokenizer(models.BPE(vocab={'Ã': 0, '©': 1, 'x': 2}, merges=[]))
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens([AddedToken('<stop>', special=True)])
        stream = TextStream(tokenizer)
        self.assertIsNone(stream.push(0))
        with self.assertRaises(ValueError): stream.push(99)
        self.assertEqual(stream.push(1), 'é')
        self.assertEqual(stream.push(2), 'x')
        self.assertIsNone(stream.push(tokenizer.token_to_id('<stop>')))

    def test_invalid_request_does_not_reset_existing_session(self):
        session = Session()
        session.reset()
        session.step(2)
        for key, value in [('prompt', [True]), ('capacity', 8), ('repetitions', 0), ('warmups', 11)]:
            workload = self.workload()
            workload[key] = value
            with self.assertRaises(ValueError): sample(session, workload, 'Greedy', Text(), np)
            self.assertEqual(session.tokens, [2])
        workload = self.workload()
        workload['fixed_output'] = [1] * 30
        with self.assertRaises(ValueError): validate_workload(workload)


if __name__ == '__main__': unittest.main()
