import Chart from "chart.js/auto";
import "./style.css";
import data from "./results.json";
import { sources } from "./sources.js";

const $ = (id) => document.getElementById(id);
const palette = ["#287354", "#447aab", "#b66b39", "#85733b", "#775c85"];
const familyNames = {
  "llama-2": "Llama 2",
  "llama-3": "Llama 3",
  "gpt-oss": "GPT-OSS",
  "qwen-3": "Qwen3",
  "gemma-4": "Gemma 4",
};
const familyColor = Object.fromEntries(
  Object.keys(familyNames).map((name, i) => [name, palette[i]]),
);
const shortModel = (name) =>
  name
    .replace(/-2507|-Instruct|-chat-hf|-it/g, "")
    .replace(/^gemma/, "Gemma")
    .replace(/^gpt-oss/, "GPT-OSS");
const sci = (n) => (n === 0 ? "0" : Number(n).toExponential(2));
const metricName = (metric) =>
  ({ bi: "BI per request", qnbi: "QNBI", quality: "Answer quality" })[metric];
const unit = (metric) =>
  metric === "quality"
    ? "Accuracy (%)"
    : metric === "qnbi"
      ? "species·year / request / unit quality"
      : "species·year / request";
const charts = {};
Chart.defaults.font.family = "Arial, Helvetica, sans-serif";
Chart.defaults.color = "#59675f";
Chart.defaults.animation = matchMedia("(prefers-reduced-motion: reduce)")
  .matches
  ? false
  : { duration: 250 };

function table(id, headings, rows) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  headings.forEach((text) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = text;
    head.append(cell);
  });
  const body = table.createTBody();
  rows.forEach((row) => {
    const tr = body.insertRow();
    row.forEach((text) => {
      tr.insertCell().textContent = text;
    });
  });
  $(id).replaceChildren(table);
}

function valuesTable(id, rows) {
  table(
    id,
    [
      "Model",
      "GPU",
      "BI / request (species·yr)",
      "QNBI (species·yr / request / quality)",
      "Quality (%)",
    ],
    rows.map((r) => [
      r.model,
      r.gpu,
      sci(r.bi),
      sci(r.qnbi),
      (r.quality * 100).toFixed(2),
    ]),
  );
}

function render(
  id,
  labels,
  datasets,
  {
    horizontal = true,
    log = false,
    percent = false,
    legend = false,
    tooltip,
  } = {},
) {
  const values = datasets.flatMap((dataset) => dataset.data).filter((value) => Number.isFinite(value) && value > 0);
  // Logarithmic bars start at the axis minimum, which must be below every value.
  const logMin = log && values.length ? 10 ** (Math.ceil(Math.log10(Math.min(...values))) - 1) / 10 : undefined;
  const valueAxis = horizontal ? "x" : "y";
  const config = {
    type: "bar",
    data: {
      labels,
      datasets: datasets.map((d) => ({
        borderRadius: 2,
        maxBarThickness: 24,
        ...d,
      })),
    },
    options: {
      responsive: true,
      animation: false,
      maintainAspectRatio: false,
      indexAxis: horizontal ? "y" : "x",
      interaction: { mode: "nearest", intersect: true },
      plugins: {
        legend: {
          display: legend,
          position: "bottom",
          labels: { boxWidth: 12, padding: 18 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              `${ctx.dataset.label}: ${percent ? ctx.raw.toFixed(2) + "%" : sci(ctx.raw)}`,
            afterLabel: tooltip || (() => ""),
          },
        },
      },
      scales: {
        [valueAxis]: {
          type: log ? "logarithmic" : "linear",
          beginAtZero: !log,
          ...(log ? { min: logMin } : {}),
          ...(percent ? { max: 100 } : {}),
          grid: { color: "#edf0ed" },
          title: { display: false },
          ticks: {
            maxTicksLimit: 6,
            callback: (v) => (percent ? `${v}%` : sci(v)),
            font: { size: 11 },
          },
        },
        [horizontal ? "y" : "x"]: {
          grid: { display: false },
          ticks: {
            autoSkip: false,
            font: { size: matchMedia("(max-width: 500px)").matches ? 10 : 12 },
          },
        },
      },
    },
  };
  const chart = charts[id];
  if (chart) {
    chart.stop();
    chart.data = config.data;
    chart.options = config.options;
    chart.resize();
    chart.update("none");
  } else {
    charts[id] = new Chart($(id), config);
  }
}

function composition() {
  const lifecycle = $("composition-view").value === "lifecycle";
  $("horizon-label").hidden = lifecycle;
  const horizon = $("horizon").value;
  const rows = lifecycle
    ? data.lifecycle
    : data.midpoint.filter((r) => r.horizon === Number(horizon));
  const names = {
    operational: "Operation",
    manufacturing: "Manufacturing",
    transportation: "Transportation",
    recycling: "End of life",
    GWP: "Global warming",
    WC: "Water consumption",
    AP: "Acidification",
    POFP: "Ozone formation",
    Others: "Other pathways",
  };
  $("composition-caption").textContent =
    `Mean share of total BI (%) · ${lifecycle ? "100" : horizon}-year assessment horizon · H100 and A100 configurations`;
  render(
    "composition-chart",
    rows.map((r) => names[r.label]),
    [
      {
        label: "Mean contribution",
        data: rows.map((r) => r.share * 100),
        backgroundColor: palette,
      },
    ],
    { percent: true },
  );
  table(
    "composition-table",
    ["Contribution", "Mean share (%)"],
    rows.map((r) => [names[r.label], (100 * r.share).toFixed(4)]),
  );
}

data.workloads.forEach((w) => $("workload").add(new Option(w.label, w.key)));
Object.entries(familyNames).forEach(([key, name]) =>
  $("model-family").add(new Option(name, key)),
);

function models() {
  const workload = data.workloads.find((w) => w.key === $("workload").value);
  const availableFamilies = new Set(workload.rows.map((r) => r.family));
  for (const option of $("model-family").options) {
    option.disabled = option.value !== "all" && !availableFamilies.has(option.value);
  }
  if ($("model-family").selectedOptions[0].disabled) $("model-family").value = "all";
  const family = $("model-family").value;
  const rows = workload.rows.filter(
    (r) => family === "all" || r.family === family,
  );
  const metric = $("model-metric").value;
  $("model-figures").textContent =
    `FIGURES ${workload.figures.replace(", ", " & ")}`;
  $("model-caption").textContent =
    `${metricName(metric)} · ${unit(metric)} · ${rows.length} models · ${$("log-scale").checked ? "logarithmic" : "linear"} scale`;
  $("model-frame").style.setProperty(
    "--chart-height",
    `${Math.max(260, rows.length * 32 + 65)}px`,
  );
  render(
    "model-chart",
    rows.map((r) => shortModel(r.model)),
    [
      {
        label: metricName(metric),
        data: rows.map((r) => r[metric]),
        backgroundColor: rows.map((r) => familyColor[r.family]),
      },
    ],
    {
      log: $("log-scale").checked,
      tooltip: (ctx) => {
        const r = rows[ctx.dataIndex];
        return [
          `${r.gpu} · ${r.moe ? "MoE" : "Dense"} · ${r.size}B total parameters`,
          `Quality: ${(r.quality * 100).toFixed(2)}%`,
        ];
      },
    },
  );
  valuesTable("model-table", rows);
}

function reasoning() {
  const metric = $("reasoning-metric").value;
  const sizes = [...new Set(data.reasoning.map((r) => r.size))];
  const percent = metric === "quality";
  const sets = ["Instruct", "Thinking"].map((mode, i) => ({
    label: mode,
    backgroundColor: palette[i],
    data: sizes.map(
      (size) =>
        data.reasoning.find((r) => r.mode === mode && r.size === size)[metric] *
        (percent ? 100 : 1),
    ),
  }));
  $("reasoning-caption").textContent =
    `${metricName(metric)} · ${unit(metric)} · MMLU-Pro · H100 · ${percent ? "linear" : "logarithmic"} scale`;
  render(
    "reasoning-chart",
    sizes.map((size) => `Qwen3 ${size}`),
    sets,
    { legend: true, percent, log: !percent },
  );
  valuesTable("reasoning-table", data.reasoning);
}

function gpu() {
  const family = $("gpu-family").value;
  const rows = data.gpus[family];
  const metric = $("gpu-metric").value;
  const models = [...new Set(rows.map((r) => r.model))];
  $("gpu-figures").textContent = family === "qwen" ? "FIGURE 10" : "FIGURE 27";
  $("gpu-caption").textContent =
    `${metricName(metric)} · ${unit(metric)} · ShareGPT · logarithmic scale`;
  const sets = ["L40", "A100", "H100"].map((gpu, i) => ({
    label: gpu,
    backgroundColor: [palette[2], palette[1], palette[0]][i],
    data: models.map(
      (model) =>
        rows.find((r) => r.model === model && r.gpu === gpu)?.[metric] ?? null,
    ),
  }));
  render("gpu-chart", models.map(shortModel), sets, {
    legend: true,
    log: true,
  });
  valuesTable("gpu-table", rows);
}

for (const [ids, callback] of [
  [["composition-view", "horizon"], composition],
  [["workload", "model-metric", "model-family", "log-scale"], models],
  [["reasoning-metric"], reasoning],
  [["gpu-family", "gpu-metric"], gpu],
]) {
  ids.forEach((id) => $(id).addEventListener("change", callback));
  callback();
}

sources.forEach((group) => {
  const section = document.createElement("section");
  section.className = "source-group";
  const title = document.createElement("h3");
  title.textContent = group.group;
  section.append(title);
  group.items.forEach(([name, citation, role, url]) => {
    const row = document.createElement("div");
    row.className = "source-row";
    const identity = document.createElement("div");
    const heading = document.createElement("h4");
    heading.textContent = name;
    const meta = document.createElement("p");
    meta.className = "source-meta";
    meta.textContent = citation;
    identity.append(heading, meta);
    const description = document.createElement("p");
    description.textContent = role;
    const link = document.createElement("a");
    link.href = url;
    link.textContent = "Source ↗";
    link.setAttribute("aria-label", `Source: ${name}`);
    row.append(identity, description, link);
    section.append(row);
  });
  $("source-list").append(section);
});

$("copy-citation").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("bibtex").textContent);
    $("copy-status").textContent = "Citation copied.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($("bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    $("copy-status").textContent =
      "Citation selected. Copy using your browser.";
  }
});
