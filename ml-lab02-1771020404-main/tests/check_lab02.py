from pathlib import Path
import ast
import nbformat

required_files = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("knn_manual.py"),
    Path("data/StudentPerformanceFactors.csv"),
    Path("Lab02_KNN.ipynb"),
]

missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))

nb_path = Path("Lab02_KNN.ipynb")
nb = nbformat.read(nb_path, as_version=4)

markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

required_sections = [
    "Mission 1",
    "Mission 2",
    "Mission 3",
    "Mission 4",
    "Mission 5",
    "Mission 6",
    "Transfer Challenge",
    "Model Reflection Card",
]

missing_sections = [s for s in required_sections if s.lower() not in markdown.lower()]
if missing_sections:
    raise SystemExit("Notebook is missing sections: " + ", ".join(missing_sections))

prohibited = ["GridSearchCV", "/content/drive/", "C:/Users/", "C:\\Users\\", "/Users/"]
found_prohibited = [p for p in prohibited if p in code]
if found_prohibited:
    raise SystemExit("Prohibited pattern(s) found: " + ", ".join(found_prohibited))

# Kiểm tra feature_cols nếu có thể đọc literal list.
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
                    if isinstance(value, list) and "Exam_Score" in value:
                        raise SystemExit("Target leakage: Exam_Score must not be in feature_cols")

placeholders = ["[VIẾT", "[TRẢ LỜI", "Họ tên: ...", "MSSV: ...", "Lớp: ..."]
found = [p for p in placeholders if p.lower() in markdown.lower()]

print("✅ Required files found")
print("✅ Notebook structure is valid")
print("✅ No prohibited paths/GridSearchCV found")
print("✅ feature_cols does not contain Exam_Score (when statically detectable)")

if found:
    print("⚠️ Notebook still contains answer placeholders:")
    for item in found:
        print("  -", item)
else:
    print("✅ No common answer placeholders detected")
