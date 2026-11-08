"""Portable, SHA-bound replay of the existing sensitivity18 fit caches/30 routes.

Only stdlib is loaded during inspection. Numerical execution belongs to a
separately admitted finite process. Original imputation/inference/calibration
statements are retained; this module has no fitting or bootstrap entrypoint.
"""
from pathlib import Path, PurePosixPath
import ast
import builtins
import hashlib
import importlib.util
import json
import sys
import types

SOURCE_SHA = "afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
PATH_MODULE_SHA = "f452cf98fd5175dfaed06757c2fb7a26e431b90d39546fc40a912543e34bf002"
RUNTIME = "work/model_lock_20261022_v1/runtime.py"
FITTING_SOURCE = "work/fixed_flow_sensitivity_20261024_v1/fit.py"
STABILITY_SOURCE = "work/conditional_stability_20261024_v1/stability.py"
CALIBRATION_SOURCE = "work/ensemble_calibration_20261020_v1/ensemble_math.py"
SOURCE = PurePosixPath("outputs/v4_development_inputs_20261018")
NATIVE_SOURCES = {
    RUNTIME: "a6c98476898c5ed061a2f2913590bde54a59b42ae50f42cef8ac30b2228c209f",
    FITTING_SOURCE: "2e2e2ddd5b35146a69f6d688666f2033f77263a17bf2952e1331c61fa8fd047e",
    STABILITY_SOURCE: "a2d914f9e3f0e65c6ae10abf7a4b8466211bed0a91c90616bcbbc48ac7d284a6",
    CALIBRATION_SOURCE: "03ad1d7342bb92f31d87fdc760e3a8067649777e1bca78a7ddd5d803d31c199f",
    "work/python_six_outcomes_20261017/donor_helpers.py":
        "409b3bb4eb201a9789f257491f6dcddfbc0c807fbe8ca67d6868c22418d653e4",
    "work/python_six_outcomes_20261017/mixed_imputer.py":
        "c4235085387f4d8b367ef8662dc213d9455dfe46d0b35d8116796ea2ebcc47db",
    "work/v2_stable_imputation_20261017/stable_imputer.py":
        "48536d164f2e401265d34d6bbc565dc78aac17f203f7eae6d1c3ccca1ce219e5",
}
CLASS_MODULES = (
    ("donor_helpers", "work/python_six_outcomes_20261017/donor_helpers.py"),
    ("mixed_imputer", "work/python_six_outcomes_20261017/mixed_imputer.py"),
    ("stable_imputer", "work/v2_stable_imputation_20261017/stable_imputer.py"),
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def parse_native(path, relative):
    if sha(path) != NATIVE_SOURCES[relative]:
        raise ValueError("Native source SHA mismatch: " + relative)
    return ast.parse(Path(path).read_text(), filename=str(path))

def node_summary(nodes):
    tree = ast.Module(body=nodes, type_ignores=[])
    calls = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
    if any(call.split(".")[-1] in {
        "fit", "fit_transform", "dump", "save", "write_text", "mkdir"
    } for call in calls):
        raise ValueError("Training/writing statement in replay subset")
    return {
        "AST_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "original_lines": [[n.lineno, n.end_lineno] for n in nodes],
        "calls": calls,
    }

def inference_trees(paths):
    selected, provenance = {}, {}
    for label, relative, names in [
        ("runtime", RUNTIME, ("source_spec", "keys")),
        ("heldout", FITTING_SOURCE, ("heldout",)),
        ("calibration", CALIBRATION_SOURCE, ("apply_calibration",)),
    ]:
        parsed = parse_native(paths[relative], relative)
        nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if len(nodes) != len(names) or {n.name for n in nodes} != set(names):
            raise ValueError("Native function set mismatch")
        selected[label] = ast.Module(body=nodes, type_ignores=[])
        provenance[label] = node_summary(nodes)
    parsed = parse_native(paths[STABILITY_SOURCE], STABILITY_SOURCE)
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "fit_flow"]
    if len(functions) != 1:
        raise ValueError("Ambiguous original fit_flow source")
    matches = []
    for node in ast.walk(functions[0]):
        for _, body in ast.iter_fields(node):
            if not isinstance(body, list):
                continue
            for i, statement in enumerate(body):
                if isinstance(statement, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "allp" for t in statement.targets
                ):
                    matches.append(body[i:i+5])
    if len(matches) != 1:
        raise ValueError("Ambiguous native full-batch reload block")
    block = matches[0]
    if len(block) != 5 or not all(isinstance(n, t) for n, t in zip(
        block, (ast.Assign, ast.Assign, ast.For, ast.Assign, ast.Expr)
    )):
        raise ValueError("Native inference block structure changed")
    if ast.unparse(block[1].targets[0]) != "reloaded_filled":
        raise ValueError("Native imputation cache statement changed")
    if ast.unparse(block[3].targets[0]) != "batch":
        raise ValueError("Native ensemble statement changed")
    last = block[4].value
    if not isinstance(last, ast.Call) or ast.unparse(last.func) != "np.testing.assert_allclose":
        raise ValueError("Native reload comparison missing")
    provenance["full_batch_reload_block"] = node_summary(block)
    # The five borrowed statements remain exact original AST nodes. Only their
    # function envelope and return of the already computed vector are new.
    wrapper = ast.parse("def predict_full_batch(loaded, lock, valid, vm, raw):\n    pass\n").body[0]
    wrapper.body = block + [ast.Return(value=ast.Name(id="batch", ctx=ast.Load()))]
    selected["inference"] = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    for label, tree in selected.items():
        compile(tree, "<verified-package-sensitivity-" + label + ">", "exec")
    return selected, provenance

def prepare_runtime(package_root, manifest_path, expected_manifest_sha256):
    mapper = Path(__file__).with_name("locked_package_paths.py")
    if sha(mapper) != PATH_MODULE_SHA:
        raise ValueError("Qualified path mapper changed")
    spec = importlib.util.spec_from_file_location("_aki_sensitivity_package_paths", mapper)
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)
    resolver = mapping.LockedPackagePaths(package_root, manifest_path,
                                         expected_manifest_sha256, SOURCE_SHA)
    paths = {}
    for relative, expected in NATIVE_SOURCES.items():
        path = resolver.resolve(relative)
        if sha(path) != expected:
            raise ValueError("Package source hash mismatch")
        paths[relative] = path
    for relative in [str(SOURCE/"FEATURE_SPECS.json"), str(SOURCE/"SOURCE_LOCK.json")]:
        resolver.resolve(relative)
    trees, provenance = inference_trees(paths)
    return resolver, paths, trees, provenance

def create_native_runtime(package_root, manifest_path, expected_manifest_sha256):
    resolver, paths, trees, provenance = prepare_runtime(
        package_root, manifest_path, expected_manifest_sha256)
    import numpy as np
    import pandas as pd
    import joblib
    for name, relative in CLASS_MODULES:
        path = paths[relative]
        previous = sys.modules.get(name)
        if previous is not None:
            origin = getattr(previous, "__file__", None)
            if origin is None or Path(origin).resolve() != path.resolve():
                raise ValueError("Conflicting preloaded custom class: " + name)
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        if Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError("Custom class outside the package")

    class Reference:
        """Preserve native path expressions; bind every read to the exact map."""
        def __init__(self, reference):
            self.reference = PurePosixPath(reference)
        def __truediv__(self, child):
            return Reference(self.reference/child)
        def __str__(self):
            return str(self.reference)
        def __fspath__(self):
            return str(self.reference)
        def read_text(self, *args, **kwargs):
            return resolver.resolve(str(self.reference)).read_text(*args, **kwargs)

    def read_json(reference):
        return json.loads(resolver.resolve(str(reference)).read_text())
    def read_csv(reference, *args, **kwargs):
        return pd.read_csv(resolver.resolve(str(reference)), *args, **kwargs)
    def load_state(reference, *args, **kwargs):
        return joblib.load(resolver.resolve(str(reference)), *args, **kwargs)

    native = {"__builtins__": builtins.__dict__, "json": json, "SOURCE": Reference(SOURCE)}
    exec(compile(trees["runtime"], "<verified-package-sensitivity-runtime>", "exec"), native)
    rt = types.SimpleNamespace(source_spec=native["source_spec"], keys=native["keys"])
    namespace = {
        "__builtins__": builtins.__dict__, "np": np, "rt": rt,
        "pd": types.SimpleNamespace(read_csv=read_csv),
        "base": types.SimpleNamespace(SOURCE=Reference(SOURCE)),
    }
    for label in ["heldout", "calibration", "inference"]:
        exec(compile(trees[label], "<verified-package-sensitivity-" + label + ">", "exec"), namespace)
    return types.SimpleNamespace(
        heldout=namespace["heldout"], keys=rt.keys, source_spec=rt.source_spec,
        predict_full_batch=namespace["predict_full_batch"],
        apply_calibration=namespace["apply_calibration"],
        read_json=read_json, read_csv=read_csv, load_state=load_state,
        resolver=resolver, function_provenance=provenance,
        numerical_versions={"numpy": np.__version__, "pandas": pd.__version__,
                            "joblib": joblib.__version__},
        training_available=False,
    )
