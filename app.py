from pathlib import Path
import runpy
import sys

app_dir = Path(__file__).resolve().parent / "expense tracker"
sys.path.insert(0, str(app_dir))
runpy.run_path(str(app_dir / "app.py"), run_name="__main__")
