import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from experiments.mad.tasks import TASKS,TaskSpec,VALUE,dataset
from .baselines import predictions,summarize,audit

class ShortcutTests(unittest.TestCase):
    def test_predictions_do_not_read_answers(self):
        for task in TASKS:
            spec=TaskSpec(task)
            for ep in dataset(spec,4,"test"):
                changed=replace(ep,
                    data=bytes(VALUE+(v-VALUE+1)%spec.vocab if m else v for v,m in zip(ep.data,ep.mask)),
                    answer=tuple((v+1)%spec.vocab for v in ep.answer))
                self.assertEqual(predictions(ep,spec),predictions(changed,spec),task)

    def test_pointer_shortcut_excludes_start_but_knows_no_edges(self):
        for length,depth in ((4,1),(4,2),(8,4)):
            spec=TaskSpec("pointer_chasing",length=length,depth=depth)
            rows=dataset(spec,16,"test")
            metric=summarize(rows,spec)["uniform_nonstart_node_distribution"]
            self.assertAlmostEqual(metric["token_accuracy"],1/(length-1))

    def test_summary_arrays_are_not_experiment_objects(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"other.json").write_text(json.dumps([{"unrelated":True}]))
            self.assertEqual(audit(d),[])

if __name__=="__main__":unittest.main()
