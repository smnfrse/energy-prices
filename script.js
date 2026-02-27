// DE-LU Energy Price Forecast Dashboard

(function () {
  "use strict";

  let currentLang = "en";
  let translations = {};

  const DATA_BASE = "data/";

  // --- Data loading ---

  async function fetchJSON(filename) {
    try {
      const resp = await fetch(DATA_BASE + filename);
      if (!resp.ok) return null;
      return await resp.json();
    } catch {
      return null;
    }
  }

  async function init() {
    translations = (await fetchJSON("../translations.json")) || {};

    const [forecast, actuals, metadata, history] = await Promise.all([
      fetchJSON("forecast_latest.json"),
      fetchJSON("actuals_latest.json"),
      fetchJSON("metadata.json"),
      fetchJSON("forecast_history.json"),
    ]);

    renderForecastChart(forecast);
    renderHistoryChart(actuals, history);
    renderMetadata(metadata);
    setupLanguageToggle();
  }

  // --- Forecast bar chart ---

  function renderForecastChart(forecast) {
    const container = document.getElementById("forecast-chart");
    if (!forecast || !forecast.prices) return;

    container.innerHTML = "";

    const hours = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, "0") + ":00");
    const trace = {
      x: hours,
      y: forecast.prices,
      type: "bar",
      marker: { color: "#4361ee" },
      hovertemplate: "%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
    };

    const layout = {
      xaxis: { title: t("hour"), tickangle: -45 },
      yaxis: { title: t("unit") },
      margin: { t: 20, r: 30, b: 60, l: 60 },
      height: 320,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });

    const label = document.getElementById("forecast-date-label");
    if (label) label.textContent = t("forecast_date") + ": " + forecast.date;
  }

  // --- History: forecast vs actual ---

  function renderHistoryChart(actuals, history) {
    const container = document.getElementById("history-chart");
    if (!actuals || !actuals.days || actuals.days.length === 0) return;

    container.innerHTML = "";

    const traces = [];

    // Actual prices for recent days
    actuals.days.forEach(function (day) {
      const hours = day.prices.map(function (_, i) {
        return day.date + " " + String(i).padStart(2, "0") + ":00";
      });
      traces.push({
        x: hours,
        y: day.prices,
        type: "scatter",
        mode: "lines",
        name: t("actual") + " " + day.date,
        line: { width: 1.5 },
        hovertemplate: "%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
      });
    });

    // Overlay forecast history if available
    if (history && history.length > 0) {
      history.forEach(function (entry) {
        // Only show forecasts for dates that have actuals (for comparison)
        var matchingActual = actuals.days.find(function (d) { return d.date === entry.date; });
        if (!matchingActual) return;

        var hours = entry.prices.map(function (_, i) {
          return entry.date + " " + String(i).padStart(2, "0") + ":00";
        });
        traces.push({
          x: hours,
          y: entry.prices,
          type: "scatter",
          mode: "lines",
          name: t("forecast") + " " + entry.date,
          line: { width: 1.5, dash: "dash" },
          hovertemplate: "%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
        });
      });
    }

    if (traces.length === 0) return;

    var layout = {
      xaxis: { title: "", tickangle: -45 },
      yaxis: { title: t("unit") },
      margin: { t: 20, r: 30, b: 80, l: 60 },
      height: 350,
      legend: { orientation: "h", y: -0.3 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
    };

    Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
  }

  // --- Metadata ---

  function renderMetadata(metadata) {
    if (!metadata) return;

    document.getElementById("meta-blend-mae").textContent =
      metadata.blend_mae != null ? metadata.blend_mae.toFixed(2) + " EUR/MWh" : "\u2014";
    document.getElementById("meta-n-models").textContent =
      metadata.n_models != null ? metadata.n_models : "\u2014";
    document.getElementById("meta-categories").textContent =
      metadata.model_categories ? metadata.model_categories.join(", ") : "\u2014";

    if (metadata.last_updated) {
      var d = new Date(metadata.last_updated);
      document.getElementById("last-updated-value").textContent = d.toLocaleString();
    }

    var warning = document.getElementById("reselection-warning");
    if (metadata.needs_reselection) {
      warning.hidden = false;
    }
  }

  // --- Language toggle ---

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
