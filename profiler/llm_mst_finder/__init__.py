from .loadgen import (
    ClosedLoopLoadGenerator,
    OpenLoopLoadGenerator,
    RequestSourceExhausted,
    count_unique_request_reuse_keys,
    cycling_request_source,
    request_reuse_key,
    request_source_factory_for_reuse_policy,
    unique_request_source,
)
from .metrics_polling import PrometheusMetricsPoller, parse_prometheus_text, parse_server_metrics_sample
from .model_context import (
    ContextPolicy,
    ContextValidationReport,
    ContextValidationResult,
    ModelContextInfo,
    parse_context_policy,
    resolve_model_context_info,
    resolve_model_tokenizer_for_policy,
    validate_samples_against_context_window,
)
from .records import (
    BenchmarkMetrics,
    BottleneckResult,
    RequestRecord,
    SampleRequest,
    ScheduledRequest,
    ServerMetricSample,
    StabilityResult,
    TrialConfig,
    TrialAnalysisResult,
    TrialSummary,
    WindowSummary,
)
from .reporting import generate_report
from .request_client import RequestClient
from .search import (
    ClosedLoopScoutResult,
    InvalidSearchTrial,
    SearchConfig,
    SearchController,
    SearchConvergenceError,
    SearchError,
    SearchResult,
)
from .trial_runner import TrialArtifacts, TrialRunResult, TrialRunner
from .workload import (
    PreparedWorkload,
    generate_sample_requests,
    inspect_workload_dataset,
    load_workload_config,
    load_workload_samples,
    load_workload_samples_for_sampling_only,
    prepare_workload_for_trial,
)

try:
    from .plotting import plot_result_comparison, plot_search_results, plot_trial_windows
except ModuleNotFoundError as exc:
    if exc.name not in {"matplotlib", "matplotlib.pyplot"}:
        raise

    def _missing_plotting_dependency(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "plotting requires matplotlib, which is not installed in this environment"
        ) from exc

    plot_result_comparison = _missing_plotting_dependency
    plot_search_results = _missing_plotting_dependency
    plot_trial_windows = _missing_plotting_dependency

__all__ = [
    "BenchmarkMetrics",
    "BottleneckResult",
    "ClosedLoopLoadGenerator",
    "ClosedLoopScoutResult",
    "ContextPolicy",
    "ContextValidationReport",
    "ContextValidationResult",
    "InvalidSearchTrial",
    "OpenLoopLoadGenerator",
    "ModelContextInfo",
    "PreparedWorkload",
    "PrometheusMetricsPoller",
    "RequestClient",
    "RequestRecord",
    "RequestSourceExhausted",
    "SampleRequest",
    "SearchConfig",
    "SearchController",
    "SearchConvergenceError",
    "SearchError",
    "SearchResult",
    "ScheduledRequest",
    "ServerMetricSample",
    "StabilityResult",
    "TrialArtifacts",
    "TrialAnalysisResult",
    "TrialConfig",
    "TrialRunResult",
    "TrialRunner",
    "TrialSummary",
    "WindowSummary",
    "count_unique_request_reuse_keys",
    "cycling_request_source",
    "generate_report",
    "generate_sample_requests",
    "inspect_workload_dataset",
    "load_workload_config",
    "load_workload_samples",
    "load_workload_samples_for_sampling_only",
    "prepare_workload_for_trial",
    "plot_result_comparison",
    "plot_search_results",
    "plot_trial_windows",
    "parse_context_policy",
    "parse_prometheus_text",
    "parse_server_metrics_sample",
    "request_reuse_key",
    "request_source_factory_for_reuse_policy",
    "resolve_model_tokenizer_for_policy",
    "resolve_model_context_info",
    "unique_request_source",
    "validate_samples_against_context_window",
]
