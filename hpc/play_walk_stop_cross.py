"""Record a walk-stop-cross episode with v3.2 using the existing video backend."""
from pathlib import Path
import sys
from play_obstacle_v31 import main

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    defaults = {'--usd': str(root/'v3.2/v3.2.usd'), '--steps': '400',
                '--output-dir': str(root/'recordings/walk_stop_cross_v32'),
                '--video-backend': 'software'}
    for option,value in defaults.items():
        if not any(arg == option or arg.startswith(option+'=') for arg in sys.argv[1:]):
            sys.argv.extend((option,value))
    main()
