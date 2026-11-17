"""Original 23 protected-loader checks on all three execution paths."""
from .backend import Model
from experiments.row_parallel_affine.backend import Model as Four
from experiments.selective_record_key_credit.backend import Model as Scalar
from experiments.selective_record_key_credit import check_policy_isolation as original
def main():
    for cls in (Scalar,Four,Model):
        original.Model=cls;original.main()
if __name__=='__main__':main()
