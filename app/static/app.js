"use strict";

(function () {
  const QUERY_ENDPOINT = "/api/query";
  const SUMMARY_ENDPOINT = "/api/summary";
  const SOURCE_ENDPOINT = "/api/source";
  const HEALTH_ENDPOINT = "/health";
  const MAX_QUESTION_LENGTH = 500;
  // The only external link ever rendered is the official INEGI source page.
  const APPROVED_SOURCE_HOST = "www.inegi.org.mx";

  const MODE_LABELS = {
    deterministic: "Exact calculation",
    generated: "Exact calculation · explained by Gemini",
    extractive: "Semantic retrieval · dataset facts",
    insufficient_data: "Not enough data for this question",
    unsupported: "Unsupported question",
  };
  const BASIS_LABELS = {
    complete_configured_dataset: "Basis: complete configured dataset",
    filtered_subset: "Basis: filtered subset of the configured dataset",
    partial_dataset: "Basis: partial dataset",
  };

  const $ = function (id) {
    return document.getElementById(id);
  };
  const form = $("query-form");
  const input = $("question");
  const submitButton = $("submit");
  const formMessage = $("form-message");
  const result = $("result");
  const modeBadge = $("mode-badge");
  const basisBadge = $("basis-badge");
  const answer = $("answer");
  const filtersBlock = $("filters-block");
  const filtersList = $("filters");
  const metricsBlock = $("metrics-block");
  const metricsBody = $("metrics-table").querySelector("tbody");
  const evidenceBlock = $("evidence-block");
  const evidenceList = $("evidence");
  const methodologyBlock = $("methodology-block");
  const methodology = $("methodology");
  const resultAttribution = $("result-attribution");
  const serviceStatus = $("service-status");

  let inFlight = false;

  function clearChildren(element) {
    while (element.firstChild) element.removeChild(element.firstChild);
  }

  function isApprovedUrl(value) {
    if (typeof value !== "string") return false;
    try {
      const url = new URL(value);
      return url.protocol === "https:" && url.hostname === APPROVED_SOURCE_HOST;
    } catch (_error) {
      return false;
    }
  }

  function showMessage(text) {
    formMessage.textContent = text;
    formMessage.hidden = false;
  }

  function hideMessage() {
    formMessage.textContent = "";
    formMessage.hidden = true;
  }

  function setBusy(busy) {
    inFlight = busy;
    input.disabled = busy;
    submitButton.disabled = busy;
    submitButton.textContent = busy ? "Analyzing…" : "Ask";
    result.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function formatNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US") : String(value);
  }

  function tag(text) {
    const item = document.createElement("li");
    item.className = "tag";
    item.textContent = text;
    return item;
  }

  function addRow(label, value) {
    const row = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    const td = document.createElement("td");
    td.textContent = value;
    row.appendChild(th);
    row.appendChild(td);
    metricsBody.appendChild(row);
  }

  function renderMetrics(metrics) {
    clearChildren(metricsBody);
    let rows = 0;
    Object.keys(metrics || {}).forEach(function (key) {
      const value = metrics[key];
      if (Array.isArray(value)) {
        value.forEach(function (entry) {
          if (entry && typeof entry === "object" && "label" in entry && "count" in entry) {
            const share = typeof entry.share_pct === "number" ? " (" + entry.share_pct + "%)" : "";
            addRow(String(entry.label), formatNumber(entry.count) + share);
            rows += 1;
          } else if (entry && typeof entry === "object" && "name" in entry) {
            addRow(String(entry.name), [entry.activity_label, entry.locality, entry.borough, entry.stratum_label].filter(Boolean).join(" · "));
            rows += 1;
          }
        });
      } else if (value && typeof value === "object") {
        // Either a borough block ({total, by_category, share_pct}) or a flat map of numbers.
        const isBoroughBlock = "total" in value && value.by_category && typeof value.by_category === "object";
        if (isBoroughBlock) {
          addRow(key + " · total", formatNumber(value.total));
          rows += 1;
          Object.keys(value.by_category).forEach(function (cat) {
            const share = value.share_pct && typeof value.share_pct[cat] === "number" ? " (" + value.share_pct[cat] + "%)" : "";
            addRow(key + " · " + cat, formatNumber(value.by_category[cat]) + share);
            rows += 1;
          });
        } else {
          Object.keys(value).forEach(function (inner) {
            const innerValue = value[inner];
            if (typeof innerValue === "number" || typeof innerValue === "string") {
              addRow(key + " · " + inner, formatNumber(innerValue));
              rows += 1;
            }
          });
        }
      } else if (typeof value === "number" || typeof value === "string") {
        addRow(key.replace(/_/g, " "), formatNumber(value));
        rows += 1;
      }
    });
    metricsBlock.hidden = rows === 0;
  }

  function renderResponse(data) {
    const mode = typeof data.mode === "string" && MODE_LABELS[data.mode] ? data.mode : "deterministic";
    modeBadge.textContent = MODE_LABELS[mode];
    modeBadge.dataset.mode = mode;
    const basis = data.scope && BASIS_LABELS[data.scope.basis] ? data.scope.basis : null;
    basisBadge.textContent = basis ? BASIS_LABELS[basis] : "";
    basisBadge.hidden = !basis;
    basisBadge.dataset.basis = basis || "";
    answer.textContent = typeof data.answer === "string" ? data.answer : "";

    clearChildren(filtersList);
    const filters = data.filters || {};
    const filterTags = []
      .concat((filters.boroughs || []).map(function (b) { return "Borough: " + b; }))
      .concat((filters.categories || []).map(function (c) { return "Category: " + c; }))
      .concat((filters.strata || []).map(function (s) { return "Employment range: " + s; }));
    filterTags.forEach(function (text) { filtersList.appendChild(tag(text)); });
    filtersBlock.hidden = filterTags.length === 0;

    renderMetrics(mode === "deterministic" || mode === "generated" ? data.metrics : {});

    clearChildren(evidenceList);
    const evidence = Array.isArray(data.evidence) ? data.evidence : [];
    evidence.forEach(function (item) {
      if (!item || typeof item !== "object") return;
      const li = document.createElement("li");
      li.className = "source-card";
      const head = document.createElement("div");
      head.className = "source-title";
      const kind = document.createElement("span");
      kind.textContent = item.kind === "aggregate" ? "Calculated aggregate" : "Sample establishment record";
      head.appendChild(kind);
      if (typeof item.score === "number") {
        const score = document.createElement("span");
        score.className = "source-score";
        score.textContent = "similarity " + item.score.toFixed(2);
        head.appendChild(score);
      }
      const text = document.createElement("p");
      text.className = "source-excerpt";
      text.textContent = typeof item.text === "string" ? item.text : "";
      li.appendChild(head);
      li.appendChild(text);
      evidenceList.appendChild(li);
    });
    evidenceBlock.hidden = evidence.length === 0;

    methodology.textContent = typeof data.methodology === "string" ? data.methodology : "";
    resultAttribution.textContent = typeof data.attribution === "string" ? data.attribution : "";
    const sourceLink = $("source-link");
    if (data.source && isApprovedUrl(data.source.url)) sourceLink.href = data.source.url;
    methodologyBlock.hidden = false;
    result.hidden = false;
  }

  async function readErrorMessage(response) {
    const fallback = "The request could not be completed (HTTP " + response.status + ").";
    try {
      const payload = await response.json();
      if (payload && payload.error && typeof payload.error.message === "string") return payload.error.message;
    } catch (_error) {
      // Non-JSON error body; use the generic message.
    }
    return fallback;
  }

  async function submitQuestion(rawQuestion) {
    const question = rawQuestion.replace(/\s+/g, " ").trim();
    hideMessage();
    if (!question) {
      showMessage("Please enter a question before submitting.");
      input.focus();
      return;
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      showMessage("Questions are limited to " + MAX_QUESTION_LENGTH + " characters.");
      return;
    }
    if (inFlight) return;
    setBusy(true);
    try {
      const response = await fetch(QUERY_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ question: question }),
      });
      if (!response.ok) {
        showMessage(await readErrorMessage(response));
        return;
      }
      renderResponse(await response.json());
    } catch (_error) {
      showMessage("Could not reach the service. Check your connection and try again.");
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  function setList(id, items, render) {
    const list = $(id);
    clearChildren(list);
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "Not available.";
      list.appendChild(li);
      return;
    }
    items.forEach(function (item) { list.appendChild(tag(render(item))); });
  }

  async function loadSummary() {
    try {
      const response = await fetch(SUMMARY_ENDPOINT, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("summary unavailable");
      const s = await response.json();
      $("stat-total").textContent = formatNumber(s.total_establishments);
      $("stat-boroughs").textContent = formatNumber((s.boroughs || []).length);
      $("stat-categories").textContent = formatNumber((s.categories || []).length);
      $("stat-date").textContent = typeof s.retrieved_at === "string" ? s.retrieved_at.slice(0, 10) : "—";
      $("stat-complete").textContent = s.complete ? "Complete for the configured scope" : "Partial dataset";
      setList("borough-list", s.boroughs || [], function (b) { return b.name + " · " + formatNumber(b.count); });
      setList("category-list", s.categories || [], function (c) { return c.label_en + " (" + c.label_es + ") · " + formatNumber(c.count); });
      if (typeof s.attribution === "string") {
        $("attribution").textContent = s.attribution;
        $("footer-attribution").textContent = s.attribution;
      }
    } catch (_error) {
      $("stat-complete").textContent = "Dataset not loaded";
      setList("borough-list", [], String);
      setList("category-list", [], String);
    }
  }

  async function loadSource() {
    try {
      const response = await fetch(SOURCE_ENDPOINT, { headers: { Accept: "application/json" } });
      if (!response.ok) return;
      const source = await response.json();
      const note = $("completeness-note");
      note.textContent = typeof source.completeness_note === "string" ? source.completeness_note : "";
    } catch (_error) {
      // Optional detail; the summary already shows completeness.
    }
  }

  async function loadServiceStatus() {
    try {
      const response = await fetch(HEALTH_ENDPOINT, { headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (payload.status === "unavailable" || !payload.dataset_loaded) {
        serviceStatus.textContent = "Dataset unavailable";
        serviceStatus.dataset.state = "unavailable";
      } else if (payload.generation_configured && payload.status === "ok") {
        serviceStatus.textContent = "Explanations by Gemini enabled";
        serviceStatus.dataset.state = "ok";
      } else if (payload.generation_configured) {
        serviceStatus.textContent = "Gemini explanations paused · exact answers only";
        serviceStatus.dataset.state = "degraded";
      } else {
        serviceStatus.textContent = "Deterministic mode · no language model";
        serviceStatus.dataset.state = "ok";
      }
    } catch (_error) {
      serviceStatus.textContent = "Service status unknown";
      serviceStatus.dataset.state = "unavailable";
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuestion(input.value);
  });

  document.querySelectorAll(".chip[data-question]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      input.value = chip.dataset.question || "";
      submitQuestion(input.value);
    });
  });

  loadSummary();
  loadSource();
  loadServiceStatus();
})();
