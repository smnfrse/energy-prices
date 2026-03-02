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
    renderErrorChart(actuals, history);
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
      marker: { color: "#3B82F6" },
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

    // Build single continuous "Actual" trace from all days
    var actualX = [];
    var actualY = [];
    actuals.days.forEach(function (day) {
      day.prices.forEach(function (price, i) {
        actualX.push(day.date + " " + String(i).padStart(2, "0") + ":00");
        actualY.push(price);
      });
    });

    var traces = [
      {
        x: actualX,
        y: actualY,
        type: "scatter",
        mode: "lines",
        name: t("actual"),
        line: { width: 2, color: "#64748B" },
        hovertemplate: "%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
      },
    ];

    // Build single continuous "Forecast" trace from history entries
    if (history && history.length > 0) {
      var forecastX = [];
      var forecastY = [];
      // Sort by date to ensure continuity
      var sorted = history.slice().sort(function (a, b) {
        return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
      });
      sorted.forEach(function (entry) {
        entry.prices.forEach(function (price, i) {
          forecastX.push(entry.date + " " + String(i).padStart(2, "0") + ":00");
          forecastY.push(price);
        });
      });

      if (forecastX.length > 0) {
        traces.push({
          x: forecastX,
          y: forecastY,
          type: "scatter",
          mode: "lines",
          name: t("forecast"),
          line: { width: 2, color: "#3B82F6", dash: "dash" },
          hovertemplate: "%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
        });
      }
    }

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

    if (metadata.last_updated) {
      var d = new Date(metadata.last_updated);
      document.getElementById("last-updated-value").textContent = d.toLocaleString();
    }

    var warning = document.getElementById("reselection-warning");
    if (metadata.needs_reselection) {
      warning.hidden = false;
    }
  }

  // --- Daily error metrics ---

  function computeDailyErrors(actuals, history) {
    if (!actuals || !actuals.days || !history || history.length === 0) return [];

    var actualsByDate = {};
    actuals.days.forEach(function (day) {
      actualsByDate[day.date] = day.prices;
    });

    var errors = [];
    history.forEach(function (entry) {
      var actual = actualsByDate[entry.date];
      if (!actual || actual.length !== 24 || entry.prices.length !== 24) return;

      var sumAE = 0;
      var sumSE = 0;
      for (var h = 0; h < 24; h++) {
        var diff = entry.prices[h] - actual[h];
        sumAE += Math.abs(diff);
        sumSE += diff * diff;
      }
      errors.push({
        date: entry.date,
        mae: sumAE / 24,
        rmse: Math.sqrt(sumSE / 24),
      });
    });

    errors.sort(function (a, b) {
      return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
    });

    return errors;
  }

  function renderErrorChart(actuals, history) {
    var container = document.getElementById("error-chart");
    if (!container) return;

    var errors = computeDailyErrors(actuals, history);
    if (errors.length === 0) return;

    container.innerHTML = "";

    var dates = errors.map(function (e) { return e.date; });
    var maeValues = errors.map(function (e) { return e.mae; });
    var rmseValues = errors.map(function (e) { return e.rmse; });

    var traceMae = {
      x: dates,
      y: maeValues,
      type: "bar",
      name: t("mae"),
      marker: { color: "#93C5FD" },
      hovertemplate: "%{x}<br>MAE: %{y:.2f} EUR/MWh<extra></extra>",
    };

    var traceRmse = {
      x: dates,
      y: rmseValues,
      type: "bar",
      name: t("rmse"),
      marker: { color: "#EF4444" },
      hovertemplate: "%{x}<br>RMSE: %{y:.2f} EUR/MWh<extra></extra>",
    };

    var layout = {
      barmode: "group",
      xaxis: { title: "", type: "category" },
      yaxis: { title: t("unit") },
      margin: { t: 20, r: 30, b: 60, l: 60 },
      height: 280,
      legend: { orientation: "h", y: -0.3 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
    };

    Plotly.newPlot(container, [traceMae, traceRmse], layout, {
      responsive: true,
      displayModeBar: false,
    });
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
