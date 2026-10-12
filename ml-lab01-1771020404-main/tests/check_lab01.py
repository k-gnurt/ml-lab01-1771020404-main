from pathlib import Path
import nbformat

required_files = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("data/StudentPerformanceFactors.csv"),
    Path("Lab01_StudentPerformance.ipynb"),
]

missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

nb_path = Path("Lab01_StudentPerformance.ipynb")
nb = nbformat.read(nb_path, as_version=4)

markdown = "\n".join(
    cell.source for cell in nb.cells if cell.cell_type == "markdown"
)

required_sections = [
    "Mission 1",
    "Mission 2",
    "Mission 3",
    "Mission 4",
    "Mission 5",
    "Mission 6",
    "Data Readiness Report",
]

missing_sections = [s for s in required_sections if s.lower() not in markdown.lower()]
if missing_sections:
    raise SystemExit("Notebook is missing sections: " + ", ".join(missing_sections))

placeholders = ["[ĐIỀN", "[VIẾT", "TODO", "MSSV: ...", "Họ tên: ...", "Lớp: ..."]
found = [p for p in placeholders if p.lower() in markdown.lower()]

print("✅ Required files found")
print("✅ Notebook structure is valid")

if found:
    print("⚠️ Notebook still contains placeholders:")
    for item in found:
        print("  -", item)
    print("This is a warning only; complete them before submission.")
else:
    print("✅ No common placeholders detected")
