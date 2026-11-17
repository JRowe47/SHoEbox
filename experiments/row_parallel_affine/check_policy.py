"""Original 23 protected-loader checks for both execution backends."""
from .backend import Model
from experiments.selective_record_key_credit import check_policy_isolation as original
def main():
    original.main()
    original.Model=Model
    original.main()
if __name__=='__main__':main()
