from pathlib import Path
import ast
import nbformat

required_files = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("kmeans_manual.py"),
    Path("scripts/download_data.py"),
    Path("data/StudentPerformanceFactors.csv"),
    Path("Lab05_KMeans.ipynb"),
]

missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

nb = nbformat.read("Lab05_KMeans.ipynb", as_version=4)

markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

required_sections = [
    "Mission 1",
    "Mission 2",
    "Mission 3",
    "Mission 4",
    "Mission 5",
    "Mission 6",
    "Mission 7",
    "Optional Challenge",
    "Model Reflection Card",
]

missing_sections = [s for s in required_sections if s.lower() not in markdown.lower()]
if missing_sections:
    raise SystemExit("Notebook is missing sections: " + ", ".join(missing_sections))

prohibited = [
    "GridSearchCV",
    "/content/drive/",
    "C:/Users/",
    "C:\\Users\\",
    "/Users/",
]
found_prohibited = [p for p in prohibited if p in code]
if found_prohibited:
    raise SystemExit("Prohibited pattern(s) found: " + ", ".join(found_prohibited))

expected_features = {
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
}

feature_list = None
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    try:
        tree = ast.parse(cell.source)
    except SyntaxError:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "feature_cols":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        value = None
                    if isinstance(value, list):
                        feature_list = value

if feature_list is None:
    raise SystemExit("Could not find a literal feature_cols list in the notebook.")

if set(feature_list) != expected_features:
    raise SystemExit(
        "feature_cols must contain exactly the 5 required numeric features. "
        f"Found: {feature_list}"
    )

if "Exam_Score" in feature_list or "Needs_Support" in feature_list:
    raise SystemExit("Exam_Score and Needs_Support must not be clustering features.")

placeholders = [
    "[VIẾT",
    "[ĐIỀN",
    "[CHỌN",
    "Họ tên: ...",
    "MSSV: ...",
    "Lớp: ...",
]
found = [p for p in placeholders if p.lower() in markdown.lower()]

print("✅ Required files found")
print("✅ Notebook structure is valid")
print("✅ No prohibited paths/GridSearchCV found")
print("✅ feature_cols is valid and contains no outcome/label leakage")

if found:
    print("⚠️ Notebook still contains answer placeholders:")
    for item in found:
        print("  -", item)
else:
    print("✅ No common answer placeholders detected")
