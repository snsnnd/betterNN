import time
from pathlib import Path
from analyze import main
root=Path(__file__).parent
while len(list((root/'results').glob('*.json')))<50:
 time.sleep(10)
main()
