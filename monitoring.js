// Model Monitoring Dashboard

(function () {
  "use strict";

  var currentLang = "en";
  var translations = {};

  var DATA_BASE = "data/";

  var CATEGORY_COLORS = {
    catboost: "#3B82F6",
    lgbm: "#10B981",
    xgboost: "#F59E0B",
    linear: "#8B5CF6",
    blend: "#EF4444",
  };

  // --- Data loading ---

  function fetchJSON(filename) {
    return fetch(DATA_BASE + filename)
      .then(function (resp) {
        if (!resp.ok) return null;
        return resp.json();
      })
      .catch(function () {
        return null;
      });
  }

  function init() {
    fetchJSON("../translations.json")
      .then(function (t) {
        translations = t || {};
        return Promise.all([
          fetchJSON("model_errors.json"),
          fetchJSON("metadata.json"),
          fetchJSON("retrain_history.json"),
        ]);
      })
      .then(function (results) {
        var modelErrors = results[0];
        var metadata = results[1];
        var retrainHistory = results[2];

        renderModelRmseChart(modelErrors, metadata, retrainHistory);
        renderCompositionPanel(metadata);
        renderRetrainLog(retrainHistory);
        setupLanguageToggle();
      });
  }

  // --- Per-Model RMSE Trends ---

  function renderModelRmseChart(modelErrors, metadata, retrainHistory) {
    var container = document.getElementById("rmse-chart");
    if (!modelErrors || !modelErrors.dates || modelErrors.dates.length === 0) return;

    container.innerHTML = "";

    var traces = [];
    var rmse = modelErrors.rmse;
    var dates = modelErrors.dates;

    // Get model categories from metadata for coloring
    var modelCategories = {};
    if (metadata && metadata.model_details) {
      metadata.model_details.forEach(function (m) {
        modelCategories[m.name] = m.category;
      });
    }

    // Add per-model traces
    var modelNames = Object.keys(rmse).filter(function (k) { return k !== "blend"; });
    modelNames.sort();

    modelNames.forEach(function (name) {
      var category = modelCategories[name] || "catboost";
      traces.push({
        x: dates,
        y: rmse[name],
        type: "scatter",
        mode: "lines+markers",
        name: name,
        line: { width: 1.5, color: CATEGORY_COLORS[category] || "#6B7280" },
        marker: { size: 4 },
        connectgaps: false,
        hovertemplate: name + "<br>%{x}<br>RMSE: %{y:.2f}<extra></extra>",
      });
    });

    // Add blend trace (bold)
    if (rmse.blend) {
      traces.push({
        x: dates,
        y: rmse.blend,
        type: "scatter",
        mode: "lines+markers",
        name: t("forecast") || "Blend",
        line: { width: 3, color: CATEGORY_COLORS.blend },
        marker: { size: 5 },
        hovertemplate: "Blend<br>%{x}<br>RMSE: %{y:.2f}<extra></extra>",
      });
    }

    // Retrain date markers
    var shapes = [];
    if (retrainHistory && retrainHistory.length > 0) {
      retrainHistory.forEach(function (event) {
        var retrainDate = event.date.split("T")[0];
        if (dates.indexOf(retrainDate) !== -1) {
          shapes.push({
            type: "line",
            x0: retrainDate,
            x1: retrainDate,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#9CA3AF", width: 1, dash: "dash" },
          });
        }
      });
    }

    var layout = {
      xaxis: { title: "" },
      yaxis: { title: t("unit") || "EUR/MWh" },
      margin: { t: 20, r: 30, b: 60, l: 60 },
      height: 380,
      legend: { orientation: "h", y: -0.25 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      shapes: shapes,
    };

    Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
  }

  // --- Ensemble Composition ---

  function renderCompositionPanel(metadata) {
    var chartContainer = document.getElementById("composition-chart");
    var infoEl = document.getElementById("composition-info");
    if (!metadata || !metadata.model_details || metadata.model_details.length === 0) return;

    chartContainer.innerHTML = "";

    var models = metadata.model_details.slice().sort(function (a, b) {
      return b.weight - a.weight;
    });

    var names = models.map(function (m) { return m.name; });
    var weights = models.map(function (m) { return m.weight; });
    var colors = models.map(function (m) {
      return CATEGORY_COLORS[m.category] || "#6B7280";
    });
    var hoverTexts = models.map(function (m) {
      return m.name + " (" + m.category + ")"
        + "<br>Weight: " + (m.weight * 100).toFixed(1) + "%"
        + "<br>MAE: " + (m.holdout_mae != null ? m.holdout_mae.toFixed(2) : "—")
        + "<br>RMSE: " + (m.holdout_rmse != null ? m.holdout_rmse.toFixed(2) : "—");
    });

    var trace = {
      y: names,
      x: weights,
      type: "bar",
      orientation: "h",
      marker: { color: colors },
      text: weights.map(function (w) { return (w * 100).toFixed(1) + "%"; }),
      textposition: "outside",
      hovertext: hoverTexts,
      hoverinfo: "text",
    };

    var layout = {
      xaxis: { title: "", showticklabels: false, range: [0, Math.max.apply(null, weights) * 1.3] },
      yaxis: { automargin: true },
      margin: { t: 10, r: 60, b: 20, l: 120 },
      height: Math.max(200, models.length * 32 + 60),
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
    };

    Plotly.newPlot(chartContainer, [trace], layout, { responsive: true, displayModeBar: false });

    // Info line
    var parts = [];
    if (metadata.last_retrain) {
      var d = new Date(metadata.last_retrain);
      parts.push(t("last_retrain") + ": " + d.toLocaleDateString());
    }
    if (metadata.blend_mae != null) {
      parts.push("Blend MAE: " + metadata.blend_mae.toFixed(2));
    }
    if (metadata.needs_reselection) {
      parts.push("⚠ " + t("reselection_warning"));
    }
    infoEl.textContent = parts.join(" · ");
  }

  // --- Retrain Log ---

  function renderRetrainLog(retrainHistory) {
    var container = document.getElementById("retrain-log");
    if (!retrainHistory || retrainHistory.length === 0) return;

    // Show last 5 events, newest first
    var events = retrainHistory.slice(-5).reverse();

    var html = '<table class="retrain-table">';
    html += "<thead><tr>";
    html += "<th>" + t("forecast_date") + "</th>";
    html += "<th>MAE " + t("mae_change") + "</th>";
    html += "<th>%</th>";
    html += "<th>Models</th>";
    html += "</tr></thead><tbody>";

    events.forEach(function (event) {
      var dateStr = event.date.split("T")[0];
      var rowClass = event.needs_reselection ? ' class="degraded"' : "";
      html += "<tr" + rowClass + ">";
      html += "<td>" + dateStr + "</td>";
      html += "<td>" + event.old_blend_mae.toFixed(2) + " → " + event.new_blend_mae.toFixed(2) + "</td>";
      var sign = event.degradation_pct > 0 ? "+" : "";
      html += "<td>" + sign + event.degradation_pct.toFixed(1) + "%</td>";
      html += "<td>" + event.n_models + "</td>";
      html += "</tr>";
    });

    html += "</tbody></table>";
    container.innerHTML = html;
  }

  // --- Language ---

  function t(key) {
    var lang = translations[currentLang];
    return (lang && lang[key]) || key;
  }

  function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      var text = t(key);
      if (text !== key) el.textContent = text;
    });
  }

  function setupLanguageToggle() {
    var btn = document.getElementById("lang-toggle");
    btn.addEventListener("click", function () {
      currentLang = currentLang === "en" ? "de" : "en";
      btn.textContent = currentLang === "en" ? "DE" : "EN";
      applyTranslations();
    });
  }

  // --- Boot ---

  document.addEventListener("DOMContentLoaded", init);
})();
