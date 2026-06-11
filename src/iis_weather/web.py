from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import LATEST_FEATURES, PREDICTIONS_DIR, REPORTS_DIR, ROOT_DIR
from .io import load_cities, read_json, write_json
from .models import MODEL_REGISTRY_FILE, predict_next_24h
from .preprocessing import build_latest_features


APP_HTML = """
<!doctype html>
<html lang="sl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IIS vremenska napoved</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8f5;
      --ink: #17211f;
      --muted: #65716c;
      --line: #dce4df;
      --panel: #ffffff;
      --accent: #0f766e;
      --blue: #2563eb;
      --rain: #1d4ed8;
      --warm: #b45309;
      --good: #15803d;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 24px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 21px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.2fr) minmax(340px, .9fr);
      min-height: calc(100vh - 70px);
    }
    .map-wrap { padding: 22px; border-right: 1px solid var(--line); }
    .map-shell {
      position: relative;
      width: min(100%, calc((100vh - 116px) * 1000 / 889));
      aspect-ratio: 1000 / 889;
      margin: 0 auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #a9d8ec;
      overflow: hidden;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }
    .map-shell.dragging { cursor: grabbing; }
    .map-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      transform-origin: 0 0;
      will-change: transform;
    }
    .map-image,
    .map-overlay {
      position: absolute;
      inset: 0;
      display: block;
      width: 100%;
      height: 100%;
    }
    .map-image { object-fit: fill; }
    .map-overlay { pointer-events: auto; }
    .city-dot { cursor: pointer; pointer-events: auto; }
    .city-dot circle { transition: transform .14s ease, stroke-width .14s ease; transform-box: fill-box; transform-origin: center; }
    .city-dot:hover circle, .city-dot.active circle { transform: scale(1.35); stroke-width: 3; }
    .city-label {
      font-size: 14px;
      font-weight: 800;
      letter-spacing: 0;
      fill: #ffffff;
      paint-order: stroke;
      stroke: rgba(24, 33, 31, .72);
      stroke-width: 4px;
      stroke-linejoin: round;
    }
    .map-controls {
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 4;
      display: flex;
      gap: 6px;
    }
    .map-controls button {
      width: 34px;
      height: 34px;
      padding: 0;
      border-color: rgba(255, 255, 255, .72);
      background: rgba(255, 255, 255, .9);
      font-weight: 800;
      box-shadow: 0 2px 8px rgba(23, 33, 31, .18);
    }
    .map-controls .reset-map {
      width: 48px;
      font-size: 12px;
    }
    .side { display: grid; grid-template-rows: auto 1fr; min-width: 0; }
    .toolbar {
      display: flex;
      gap: 8px;
      padding: 16px;
      background: #fff;
      border-bottom: 1px solid var(--line);
      overflow-x: auto;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 11px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      white-space: nowrap;
    }
    button.active { border-color: var(--accent); background: #e7f3ef; color: #0b4f49; font-weight: 700; }
    .content { padding: 18px; overflow: auto; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 14px;
    }
    h2 { margin: 0 0 6px; font-size: 19px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 14px; color: #2f3a36; letter-spacing: 0; }
    .muted { color: var(--muted); }
    .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
    .kpi { border: 1px solid var(--line); border-radius: 7px; padding: 10px; min-width: 0; }
    .kpi strong { display: block; font-size: 18px; margin-top: 3px; }
    .forecast { display: grid; gap: 8px; }
    .row { display: grid; grid-template-columns: 78px 1fr 128px; align-items: center; gap: 10px; font-size: 13px; }
    .bar { height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--blue)); }
    .rain { color: var(--rain); font-weight: 700; }
    .rain.dry { color: var(--muted); }
    .forecast-values {
      line-height: 1.25;
      text-align: right;
    }
    .forecast-values small {
      display: block;
      margin-top: 2px;
      font-size: 11px;
      color: var(--muted);
    }
    .chart-wrap {
      overflow-x: auto;
      padding-top: 2px;
      position: relative;
    }
    .forecast-chart {
      width: 100%;
      min-width: 560px;
      height: 250px;
      display: block;
    }
    .chart-axis {
      stroke: #ccd5cf;
      stroke-width: 1;
    }
    .chart-grid {
      stroke: #edf1ee;
      stroke-width: 1;
    }
    .chart-temp {
      fill: none;
      stroke: var(--warm);
      stroke-width: 3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .chart-rain-bar { fill: rgba(37, 99, 235, .58); cursor: crosshair; }
    .chart-dot { fill: #fff; stroke: var(--warm); stroke-width: 2; cursor: crosshair; }
    .chart-label {
      fill: var(--muted);
      font-size: 11px;
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .legend-swatch {
      width: 22px;
      height: 4px;
      border-radius: 999px;
      background: var(--warm);
    }
    .legend-swatch.rain {
      height: 10px;
      background: rgba(37, 99, 235, .58);
    }
    .chart-tooltip {
      position: absolute;
      z-index: 5;
      min-width: 132px;
      padding: 8px 10px;
      border: 1px solid rgba(24, 33, 31, .16);
      border-radius: 7px;
      background: rgba(255, 255, 255, .96);
      color: var(--ink);
      box-shadow: 0 8px 24px rgba(23, 33, 31, .16);
      font-size: 12px;
      line-height: 1.45;
      pointer-events: none;
      opacity: 0;
      transform: translate(12px, -12px);
      transition: opacity .08s ease;
    }
    .chart-tooltip.visible { opacity: 1; }
    .status {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid var(--line);
    }
    .status.pass { color: var(--good); background: #eef8ef; }
    .status.warn, .status.unknown { color: var(--warm); background: #fff7ed; }
    .status.fail { color: var(--bad); background: #fef2f2; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .map-wrap { border-right: 0; border-bottom: 1px solid var(--line); }
      .map-shell { width: 100%; }
      .kpis { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Vremenska napoved evropskih prestolnic</h1>
    <span id="global-status" class="status unknown">nalagam</span>
  </header>
  <main>
    <section class="map-wrap">
      <div class="map-shell" id="map-shell">
        <div class="map-canvas" id="map-canvas">
          <img class="map-image" src="/static/europe-map.webp" alt="Zemljevid Evrope">
          <svg class="map-overlay" id="map" viewBox="0 0 1000 889" role="img" aria-label="Prestolnice na zemljevidu Evrope"></svg>
        </div>
        <div class="map-controls" aria-label="Kontrole zemljevida">
          <button type="button" id="zoom-out" title="Oddalji">-</button>
          <button type="button" id="zoom-reset" class="reset-map" title="Ponastavi">100%</button>
          <button type="button" id="zoom-in" title="Priblizaj">+</button>
        </div>
      </div>
    </section>
    <aside class="side">
      <div id="city-tabs" class="toolbar"></div>
      <div class="content">
        <section id="city-panel" class="panel"></section>
        <section class="panel">
          <h3>Napoved naslednjih 24 ur</h3>
          <div id="forecast" class="forecast"></div>
        </section>
        <section class="panel">
          <h3>Administratorski pogled</h3>
          <div id="admin"></div>
        </section>
        <section class="panel">
          <h3>Graf napovedi</h3>
          <div class="chart-legend">
            <span class="legend-item"><span class="legend-swatch"></span>Temperatura</span>
            <span class="legend-item"><span class="legend-swatch rain"></span>Padavine</span>
          </div>
          <div id="forecast-chart" class="chart-wrap"></div>
        </section>
      </div>
    </aside>
  </main>
  <script>
    let cities = [];
    let selectedCity = null;
    const mapState = { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0, originX: 0, originY: 0, moved: false };

    function statusClass(status) {
      if (status === "pass" || status === "production") return "pass";
      if (status === "fail") return "fail";
      return "warn";
    }

    function fmt(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }

    function projectPoint(city) {
      const calibratedPositions = {
        Ljubljana: { x: 446, y: 630 },
        Zagreb: { x: 473, y: 637 },
        Vienna: { x: 471, y: 586 },
        Budapest: { x: 515, y: 590 },
        Prague: { x: 427, y: 556 },
        Berlin: { x: 431, y: 520 },
        Paris: { x: 306, y: 596 },
        Rome: { x: 430, y: 704 }
      };
      if (calibratedPositions[city.city]) return calibratedPositions[city.city];
      const minLon = -20, maxLon = 53, minLat = 34.8, maxLat = 77.5;
      const lon = Number(city.longitude), lat = Number(city.latitude);
      return {
        x: ((lon - minLon) / (maxLon - minLon)) * 1000,
        y: ((maxLat - lat) / (maxLat - minLat)) * 889
      };
    }

    function labelOffset(cityName) {
      const offsets = {
        Ljubljana: [-86, 14],
        Zagreb: [12, 10],
        Vienna: [12, -8],
        Budapest: [14, 8],
        Prague: [-58, -8],
        Berlin: [12, 4],
        Paris: [-58, 6],
        Rome: [12, 24]
      };
      return offsets[cityName] || [12, 4];
    }

    function drawMap() {
      const svg = document.getElementById("map");
      svg.innerHTML = "";
      cities.forEach(city => {
        const point = projectPoint(city);
        const active = city.city === selectedCity ? " active" : "";
        const color = Number(city.latest_temperature_c) >= 20 ? "#b45309" : "#0f766e";
        const [labelX, labelY] = labelOffset(city.city);
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", "city-dot" + active);
        group.setAttribute("data-city", city.city);
        group.innerHTML = `<circle cx="${point.x}" cy="${point.y}" r="8" fill="${color}" stroke="#ffffff" stroke-width="2"></circle><text class="city-label" x="${point.x + labelX}" y="${point.y + labelY}">${city.city}</text>`;
        svg.appendChild(group);
      });
    }

    function clampMap() {
      const shell = document.getElementById("map-shell");
      if (!shell) return;
      if (mapState.scale <= 1) {
        mapState.scale = 1;
        mapState.x = 0;
        mapState.y = 0;
        return;
      }
      const minX = shell.clientWidth * (1 - mapState.scale);
      const minY = shell.clientHeight * (1 - mapState.scale);
      mapState.x = Math.min(0, Math.max(minX, mapState.x));
      mapState.y = Math.min(0, Math.max(minY, mapState.y));
    }

    function applyMapTransform() {
      clampMap();
      const canvas = document.getElementById("map-canvas");
      if (canvas) {
        canvas.style.transform = `translate(${mapState.x}px, ${mapState.y}px) scale(${mapState.scale})`;
      }
    }

    function zoomMap(multiplier, anchorX = null, anchorY = null) {
      const shell = document.getElementById("map-shell");
      if (!shell) return;
      const oldScale = mapState.scale;
      const newScale = Math.min(5, Math.max(1, oldScale * multiplier));
      if (newScale === oldScale) return;
      const px = anchorX ?? shell.clientWidth / 2;
      const py = anchorY ?? shell.clientHeight / 2;
      mapState.x = px - (px - mapState.x) * (newScale / oldScale);
      mapState.y = py - (py - mapState.y) * (newScale / oldScale);
      mapState.scale = newScale;
      applyMapTransform();
    }

    function resetMapZoom() {
      mapState.scale = 1;
      mapState.x = 0;
      mapState.y = 0;
      applyMapTransform();
    }

    function setupMapInteractions() {
      const shell = document.getElementById("map-shell");
      if (!shell) return;
      shell.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = shell.getBoundingClientRect();
        const multiplier = event.deltaY < 0 ? 1.14 : 1 / 1.14;
        zoomMap(multiplier, event.clientX - rect.left, event.clientY - rect.top);
      }, { passive: false });
      shell.addEventListener("pointerdown", (event) => {
        if (event.target.closest?.(".map-controls")) return;
        if (event.target.closest?.(".city-dot")) return;
        shell.setPointerCapture(event.pointerId);
        mapState.dragging = true;
        mapState.moved = false;
        mapState.startX = event.clientX;
        mapState.startY = event.clientY;
        mapState.originX = mapState.x;
        mapState.originY = mapState.y;
        shell.classList.add("dragging");
      });
      shell.addEventListener("pointermove", (event) => {
        if (!mapState.dragging) return;
        const dx = event.clientX - mapState.startX;
        const dy = event.clientY - mapState.startY;
        if (Math.abs(dx) + Math.abs(dy) > 3) mapState.moved = true;
        mapState.x = mapState.originX + dx;
        mapState.y = mapState.originY + dy;
        applyMapTransform();
      });
      shell.addEventListener("pointerup", (event) => {
        if (shell.hasPointerCapture(event.pointerId)) shell.releasePointerCapture(event.pointerId);
        mapState.dragging = false;
        shell.classList.remove("dragging");
        setTimeout(() => { mapState.moved = false; }, 0);
      });
      shell.addEventListener("pointercancel", () => {
        mapState.dragging = false;
        mapState.moved = false;
        shell.classList.remove("dragging");
      });
      document.getElementById("zoom-in")?.addEventListener("click", () => zoomMap(1.25));
      document.getElementById("zoom-out")?.addEventListener("click", () => zoomMap(1 / 1.25));
      document.getElementById("zoom-reset")?.addEventListener("click", resetMapZoom);
      window.addEventListener("resize", applyMapTransform);
      document.getElementById("map")?.addEventListener("click", (event) => {
        const marker = event.target.closest?.(".city-dot");
        if (!marker || mapState.moved) return;
        const cityName = marker.getAttribute("data-city");
        if (cityName) selectCity(cityName);
      });
      applyMapTransform();
    }

    function renderTabs() {
      const tabs = document.getElementById("city-tabs");
      tabs.innerHTML = "";
      cities.forEach(city => {
        const button = document.createElement("button");
        button.textContent = city.city;
        button.className = city.city === selectedCity ? "active" : "";
        button.addEventListener("click", () => selectCity(city.city));
        tabs.appendChild(button);
      });
    }

    async function selectCity(cityName) {
      selectedCity = cityName;
      renderTabs();
      drawMap();
      const city = cities.find(item => item.city === cityName);
      document.getElementById("city-panel").innerHTML = `
        <h2>${city.city}</h2>
        <div class="muted">${city.country} · zadnja meritev ${city.latest_time || "-"}</div>
        <div class="kpis">
          <div class="kpi"><span class="muted">Temperatura</span><strong>${fmt(city.latest_temperature_c)} °C</strong></div>
          <div class="kpi"><span class="muted">Padavine</span><strong>${fmt(city.latest_precipitation_mm)} mm</strong></div>
          <div class="kpi"><span class="muted">Vlaznost</span><strong>${fmt(city.latest_relative_humidity, 0)} %</strong></div>
        </div>
      `;
      await loadForecast(cityName);
    }

    async function loadForecast(cityName) {
      const target = document.getElementById("forecast");
      target.innerHTML = "<span class='muted'>Nalagam napoved...</span>";
      const response = await fetch(`/api/predict?city=${encodeURIComponent(cityName)}`);
      if (!response.ok) {
        const error = await response.json();
        target.innerHTML = `<span class="status fail">${error.error || "Napaka"}</span>`;
        return;
      }
      const payload = await response.json();
      target.innerHTML = payload.predictions.map(row => {
        const pct = Math.round(row.precipitation_probability * 100);
        const time = new Date(row.time).toLocaleString("sl-SI", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });
        const rain = row.precipitation_expected ? "dež" : "suho";
        const rainClass = row.precipitation_expected ? "rain" : "rain dry";
        return `<div class="row"><span>${time}</span><div class="bar"><span style="width:${pct}%"></span></div><span class="forecast-values"><strong>${fmt(row.temperature_c)}</strong> °C <small><span class="${rainClass}">${rain}</span> · ${fmt(row.precipitation_mm || 0, 2)} mm · ${pct}%</small></span></div>`;
      }).join("");
      renderForecastChart(payload.predictions);
    }

    function renderForecastChart(predictions) {
      const target = document.getElementById("forecast-chart");
      if (!target || !predictions?.length) return;
      const width = 640;
      const height = 250;
      const pad = { left: 44, right: 34, top: 18, bottom: 34 };
      const plotWidth = width - pad.left - pad.right;
      const plotHeight = height - pad.top - pad.bottom;
      const temps = predictions.map(row => Number(row.temperature_c));
      const rains = predictions.map(row => Number(row.precipitation_mm || 0));
      const minTemp = Math.min(-5, Math.floor(Math.min(...temps) - 1));
      const maxTemp = Math.max(35, Math.ceil(Math.max(...temps) + 1));
      const midTemp = Math.round((minTemp + maxTemp) / 2);
      const maxRain = Math.max(0.1, Math.ceil(Math.max(...rains) * 10) / 10);
      const xFor = index => pad.left + (index / Math.max(1, predictions.length - 1)) * plotWidth;
      const tempY = value => pad.top + ((maxTemp - value) / Math.max(1, maxTemp - minTemp)) * plotHeight;
      const rainHeight = value => (value / maxRain) * (plotHeight * .72);
      const tempPath = temps.map((temp, index) => `${index === 0 ? "M" : "L"} ${xFor(index).toFixed(1)} ${tempY(temp).toFixed(1)}`).join(" ");
      const tooltipFor = (row) => {
        const time = new Date(row.time).toLocaleString("sl-SI", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
        const rain = row.precipitation_expected ? "dež" : "suho";
        return `${time} | Temperatura: ${fmt(row.temperature_c, 1)} °C | Napoved: ${rain} | Padavine: ${fmt(row.precipitation_mm || 0, 2)} mm | Verjetnost: ${Math.round(row.precipitation_probability * 100)}%`;
      };
      const bars = rains.map((rain, index) => {
        const barWidth = Math.max(8, plotWidth / predictions.length * .52);
        const x = xFor(index) - barWidth / 2;
        const h = rainHeight(rain);
        const y = pad.top + plotHeight - h;
        return `<rect class="chart-rain-bar" data-tooltip="${tooltipFor(predictions[index])}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${h.toFixed(1)}" rx="3"></rect>`;
      }).join("");
      const dots = temps.map((temp, index) => `<circle class="chart-dot" data-tooltip="${tooltipFor(predictions[index])}" cx="${xFor(index).toFixed(1)}" cy="${tempY(temp).toFixed(1)}" r="4"></circle>`).join("");
      const labels = predictions.filter((_, index) => index % 4 === 0).map((row, index) => {
        const realIndex = index * 4;
        const date = new Date(row.time);
        const label = date.toLocaleTimeString("sl-SI", { hour: "2-digit", minute: "2-digit" });
        return `<text class="chart-label" x="${xFor(realIndex).toFixed(1)}" y="${height - 10}" text-anchor="middle">${label}</text>`;
      }).join("");
      target.innerHTML = `
        <svg class="forecast-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Graf temperature in padavin">
          <line class="chart-grid" x1="${pad.left}" y1="${pad.top}" x2="${width - pad.right}" y2="${pad.top}"></line>
          <line class="chart-grid" x1="${pad.left}" y1="${pad.top + plotHeight / 2}" x2="${width - pad.right}" y2="${pad.top + plotHeight / 2}"></line>
          <line class="chart-axis" x1="${pad.left}" y1="${pad.top + plotHeight}" x2="${width - pad.right}" y2="${pad.top + plotHeight}"></line>
          <line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotHeight}"></line>
          ${bars}
          <path class="chart-temp" d="${tempPath}"></path>
          ${dots}
          <text class="chart-label" x="8" y="${pad.top + 4}">${maxTemp} °C</text>
          <text class="chart-label" x="8" y="${pad.top + plotHeight / 2 + 4}">${midTemp} °C</text>
          <text class="chart-label" x="8" y="${pad.top + plotHeight}">${minTemp} °C</text>
          <text class="chart-label" x="${width - 8}" y="${pad.top + 4}" text-anchor="end">${fmt(maxRain, 1)} mm</text>
          ${labels}
        </svg>
        <div class="chart-tooltip" id="chart-tooltip"></div>
      `;
      const tooltip = target.querySelector("#chart-tooltip");
      target.querySelectorAll("[data-tooltip]").forEach((element) => {
        element.addEventListener("mousemove", (event) => {
          const rect = target.getBoundingClientRect();
          tooltip.innerHTML = element.getAttribute("data-tooltip").replaceAll(" | ", "<br>");
          tooltip.style.left = `${event.clientX - rect.left + target.scrollLeft}px`;
          tooltip.style.top = `${event.clientY - rect.top + target.scrollTop}px`;
          tooltip.classList.add("visible");
        });
        element.addEventListener("mouseleave", () => {
          tooltip.classList.remove("visible");
        });
      });
    }

    async function loadAdmin() {
      const response = await fetch("/api/admin");
      const payload = await response.json();
      const validation = payload.validation || {};
      const monitoring = payload.monitoring || {};
      const evaluation = payload.evaluation || {};
      const registry = payload.registry || {};
      document.getElementById("global-status").textContent = validation.status || "neznano";
      document.getElementById("global-status").className = `status ${statusClass(validation.status)}`;
      document.getElementById("admin").innerHTML = `
        <p><span class="status ${statusClass(validation.status)}">podatki: ${validation.status || "-"}</span>
        <span class="status ${statusClass(monitoring.status)}">produkcija: ${monitoring.status || "-"}</span></p>
        <p class="muted">Mesta: ${validation.summary?.cities || "-"} · vrstice: ${validation.summary?.rows || "-"}</p>
        <p class="muted">Temp. MAE: ${fmt(evaluation.metrics?.temperature_regression?.mae, 2)} · Padavine F1: ${fmt(evaluation.metrics?.precipitation_classification?.f1, 2)}</p>
        <p class="muted">Aktivna verzija modela: ${registry.active_version || "-"}</p>
      `;
    }

    async function boot() {
      const response = await fetch("/api/cities");
      const payload = await response.json();
      cities = payload.cities;
      selectedCity = cities[0]?.city;
      renderTabs();
      drawMap();
      if (selectedCity) await selectCity(selectedCity);
      await loadAdmin();
    }
    setupMapInteractions();
    boot();
  </script>
</body>
</html>
"""


def _json_bytes(payload: Any, status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def _cities_payload() -> dict[str, Any]:
    cities = load_cities()
    if LATEST_FEATURES.exists():
        latest = pd.read_csv(LATEST_FEATURES, parse_dates=["time"])
    else:
        latest = build_latest_features()
    latest = latest[
        ["city", "time", "temperature_c", "precipitation_mm", "relative_humidity", "wind_speed_kmh"]
    ].rename(
        columns={
            "time": "latest_time",
            "temperature_c": "latest_temperature_c",
            "precipitation_mm": "latest_precipitation_mm",
            "relative_humidity": "latest_relative_humidity",
            "wind_speed_kmh": "latest_wind_speed_kmh",
        }
    )
    merged = cities.merge(latest, on="city", how="left")
    merged["latest_time"] = merged["latest_time"].astype(str)
    return {"cities": merged.to_dict(orient="records")}


class WeatherHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            status, body, content_type = self._route(parsed.path, parse_qs(parsed.query))
        except Exception as exc:  # pragma: no cover
            status, body, content_type = _json_bytes({"error": str(exc)}, status=500)
        self._send(status, body, content_type)

    def _route(self, path: str, query: dict[str, list[str]]) -> tuple[int, bytes, str]:
        if path in {"/", "/index.html"}:
            return 200, APP_HTML.encode("utf-8"), "text/html; charset=utf-8"
        if path == "/health":
            return _json_bytes({"status": "ok", "service": "iis-weather"})
        if path == "/static/europe-map.webp":
            image_path = ROOT_DIR / "src" / "app" / "static" / "europe-map.webp"
            if image_path.exists():
                return 200, image_path.read_bytes(), "image/webp"
            return _json_bytes({"error": "Map image not found."}, status=404)
        if path == "/api/cities":
            return _json_bytes(_cities_payload())
        if path == "/api/predict":
            city = (query.get("city") or [""])[0]
            if not city:
                return _json_bytes({"error": "Missing city query parameter."}, status=400)
            payload = predict_next_24h(city)
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            write_json(PREDICTIONS_DIR / f"{city.lower().replace(' ', '_')}_latest.json", payload)
            return _json_bytes(payload)
        if path == "/api/admin":
            return _json_bytes(
                {
                    "validation": read_json(REPORTS_DIR / "data_validation.json", default={}),
                    "evaluation": read_json(REPORTS_DIR / "model_evaluation.json", default={}),
                    "monitoring": read_json(REPORTS_DIR / "production_monitoring.json", default={}),
                    "registry": read_json(MODEL_REGISTRY_FILE, default={}),
                }
            )
        return _json_bytes({"error": "Not found."}, status=404)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), WeatherHandler)
    print(f"Serving IIS weather app on http://{host}:{port}")
    server.serve_forever()
