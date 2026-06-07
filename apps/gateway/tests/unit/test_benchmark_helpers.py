import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[4]
    benchmark_path = repo_root / "infra" / "docker" / "benchmark" / "respawn_benchmark.py"
    spec = importlib.util.spec_from_file_location("respawn_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tag_filters_select_and_skip_cases():
    benchmark = _load_benchmark_module()
    case = benchmark.BenchmarkCase("sample", {"core", "state"}, {"feature"}, lambda state: "ok")

    assert benchmark.selected_case_skip_reason(case, {"core"}, set()) is None
    assert benchmark.selected_case_skip_reason(case, {"streaming"}, set()) == "no tags in include filter streaming"
    assert benchmark.selected_case_skip_reason(case, set(), {"state"}) == "excluded tag(s): state"


def test_manifest_coverage_reports_missing_supported_features():
    benchmark = _load_benchmark_module()
    cases = [
        benchmark.BenchmarkCase("case.covered", {"core"}, {"feature.covered"}, lambda state: "ok"),
        benchmark.BenchmarkCase("case.wrong_feature", {"core"}, {"feature.other"}, lambda state: "ok"),
    ]
    manifest = {
        "features": [
            {
                "id": "feature.covered",
                "status": "supported",
                "benchmark_required": True,
                "benchmark_case": "case.covered",
            },
            {
                "id": "feature.missing_case",
                "status": "supported",
                "benchmark_required": True,
                "benchmark_case": "case.missing",
            },
            {
                "id": "feature.wrong_feature",
                "status": "supported",
                "benchmark_required": True,
                "benchmark_case": "case.wrong_feature",
            },
            {
                "id": "feature.unsupported",
                "status": "unsupported",
                "benchmark_required": True,
                "benchmark_case": "case.missing",
            },
        ]
    }

    coverage = benchmark.manifest_coverage(manifest, cases)

    assert coverage["covered_supported_features"] == ["feature.covered"]
    assert coverage["missing_supported_features"] == [
        {"id": "feature.missing_case", "benchmark_case": "case.missing"},
        {"id": "feature.wrong_feature", "benchmark_case": "case.wrong_feature"},
    ]


def test_compatibility_report_lists_skipped_surfaces():
    benchmark = _load_benchmark_module()
    state = benchmark.BenchmarkState()
    state.compatibility_manifest = {
        "features": [
            {
                "id": "feature.supported",
                "category": "endpoint",
                "surface": "POST /v1/responses",
                "status": "supported",
                "benchmark_case": "responses.blocking",
                "tags": ["core"],
            },
            {
                "id": "feature.unsupported",
                "category": "endpoint",
                "surface": "POST /v1/responses/compact",
                "status": "unsupported",
                "benchmark_case": None,
                "tags": ["state"],
            },
        ],
        "summary": {},
    }
    state.cases.append(
        benchmark.CaseResult(
            name="responses.blocking",
            ok=True,
            latency_ms=0.0,
            status="skipped",
            tags=["core"],
            feature_ids=["feature.supported"],
            skip_reason="filtered",
        )
    )

    report = benchmark.compatibility_report(state)

    assert [surface["id"] for surface in report["surfaces"]["supported"]] == ["feature.supported"]
    assert [surface["id"] for surface in report["surfaces"]["unsupported"]] == ["feature.unsupported"]
    assert [surface["id"] for surface in report["surfaces"]["skipped"]] == ["feature.supported"]
