"""Exactly the original twenty A9 tests for each of the three executions."""
import sys
from experiments.row_parallel_affine import check_python as prior
from . import backend

def main():
    if sys.argv[1]=='a11':prior.backend=backend;sys.argv[1]='row_parallel'
    elif sys.argv[1]=='a10':sys.argv[1]='row_parallel'
    elif sys.argv[1]!='scalar':raise ValueError(sys.argv[1])
    prior.main()
if __name__=='__main__':main()
