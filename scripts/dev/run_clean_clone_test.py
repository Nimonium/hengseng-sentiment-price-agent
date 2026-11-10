import os
import shutil
import subprocess
import tempfile
from pathlib import Path

repo_dir = Path(".").resolve()
temp_dir = Path(tempfile.gettempdir()) / "hengseng_clean_clone_test"

if temp_dir.exists():
    shutil.rmtree(temp_dir, ignore_errors=True)

print(f"Cloning repository {repo_dir} into {temp_dir}...")
subprocess.run(["git", "clone", str(repo_dir), str(temp_dir)], check=True)

env_in_clone = (temp_dir / ".env").exists()
print(f"Checking if .env exists in cloned repo: {env_in_clone}")
assert not env_in_clone, "FAIL: .env leaked into git clone!"
print("CONFIRMED: .env does NOT exist in clean clone.")

# 1. Run pytest in clean clone
print("\n" + "=" * 60)
print("RUNNING PYTEST IN CLEAN CLONE")
print("=" * 60)
p_res = subprocess.run(["pytest", "-v", "tests/"], cwd=str(temp_dir), capture_output=True, text=True)
print(p_res.stdout)
if p_res.stderr:
    print("Pytest stderr:", p_res.stderr)
assert p_res.returncode == 0, f"Pytest failed with code {p_res.returncode}"

# 2. Run main.py with --use-sample in clean clone
print("\n" + "=" * 60)
print("RUNNING MAIN.PY --RUN ALL --USE-SAMPLE IN CLEAN CLONE (NO .ENV)")
print("=" * 60)
m_res = subprocess.run(["python", "main.py", "--run", "all", "--use-sample"], cwd=str(temp_dir), capture_output=True, text=True)
print(m_res.stdout)
if m_res.stderr:
    print("Main stderr:", m_res.stderr)
assert m_res.returncode == 0, f"Pipeline failed with code {m_res.returncode}"

# 3. Check report.html exists and was regenerated
report_file = temp_dir / "report" / "report.html"
print(f"\nReport exists: {report_file.exists()}, size: {report_file.stat().st_size / 1024:.1f} KB")
assert report_file.exists() and report_file.stat().st_size > 10000, "Report not regenerated properly!"

# Cleanup temp dir
shutil.rmtree(temp_dir, ignore_errors=True)
print("\nClean clone test completed and cleaned up successfully!")
