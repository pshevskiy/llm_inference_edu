"""Public XML reference resolution and target filtering, no Apple runtime."""
from pathlib import Path
import tempfile
import unittest
from analyze_coreml_trace import analyze


class Contracts(unittest.TestCase):
    def test_duplicate_signpost_views_pair_once(self):
        toc = '<trace-toc><run><info><target><process pid="7" return-exit-status="0"/></target></info><data><table schema="os-signpost"/><table schema="os-signpost"/></data></run></trace-toc>'
        rows = '''<row><event-time>100</event-time><process><pid>7</pid></process><subsystem>org.llm-inference.t19</subsystem><signpost-name>Prediction</signpost-name><event-type>Begin</event-type><os-signpost-identifier>1</os-signpost-identifier></row>
          <row><event-time>200</event-time><process><pid>7</pid></process><subsystem>org.llm-inference.t19</subsystem><signpost-name>Prediction</signpost-name><event-type>End</event-type><os-signpost-identifier>1</os-signpost-identifier></row>'''
        profile = f'<trace-query-result><node xpath="//run[1]/data[1]/table[1]">{rows}</node><node xpath="//run[1]/data[1]/table[2]">{rows}</node></trace-query-result>'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'toc.xml').write_text(toc)
            (root / 'profile.xml').write_text(profile)
            result = analyze(root / 'toc.xml', root / 'profile.xml')
        self.assertEqual(result['unique_control_signpost_events'], 2)
        self.assertEqual(result['control_prediction_intervals_ns'], [[100, 200]])

    def test_referenced_stacks_sentinel_and_unrelated_process(self):
        toc = '<trace-toc><run><info><target><process pid="7" return-exit-status="0"/></target></info><data><table schema="time-profile"/></data></run></trace-toc>'
        profile = '''<trace-query-result><node xpath="//run[1]/data[1]/table[1]">
          <row><sample-time>100</sample-time><process id="p"><pid>7</pid></process>
            <tagged-backtrace id="b"><frame id="f" name="MLModel predictionFromFeatures:usingState:"/>
            <frame name="BNNSGraphContextExecute_v2"/><frame name="E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync()"/></tagged-backtrace></row>
          <row><sample-time>200</sample-time><process ref="p"/><tagged-backtrace ref="b"/></row>
          <row><sample-time>300</sample-time><process ref="p"/><tagged-backtrace><frame ref="f"/></tagged-backtrace></row>
          <row><sample-time>400</sample-time><process ref="p"/><sentinel/></row>
          <row><sample-time>500</sample-time><process><pid>8</pid></process><tagged-backtrace ref="b"/></row>
        </node></trace-query-result>'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'toc.xml').write_text(toc)
            (root / 'profile.xml').write_text(profile)
            result = analyze(root / 'toc.xml', root / 'profile.xml')
        self.assertEqual(result['target_cpu_samples'], 4)
        self.assertEqual(result['coreml_prediction_samples'], 3)
        self.assertEqual(result['bnns_cpu_prediction_samples'], 2)
        self.assertEqual(result['target_samples_without_backtrace'], 1)
        self.assertEqual(result['bnns_sample_time_range_ns'], [100, 200])


if __name__ == '__main__': unittest.main()
