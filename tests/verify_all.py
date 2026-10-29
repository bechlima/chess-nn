"""Run all phase verifications in sequence."""
import subprocess, sys

phases = [1, 2, 3, 4, 5, 6, 7, 8, 9]
failed = []

for phase in phases:
    print(f"\n--- Verifying Phase {phase} ---")
    result = subprocess.run(
        [sys.executable, f"tests/verify_phase{phase}.py"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"FAILED:\n{result.stdout}\n{result.stderr}")
        failed.append(phase)

print("\n" + "="*40)
if failed:
    print(f"FAILED phases: {failed}")
    sys.exit(1)
else:
    print(f"ALL {len(phases)} PHASES PASSED")
