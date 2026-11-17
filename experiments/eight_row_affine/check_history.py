"""Original fourteen historical checkpoint controls, explicit import roots."""
from experiments.e8_public import verify_history as original
original.WORKER='from pathlib import Path\nimport shoe_multiscale\nassert Path(shoe_multiscale.__file__).resolve().parent==Path.cwd().resolve()\n'+original.WORKER
if __name__=='__main__':original.main()
