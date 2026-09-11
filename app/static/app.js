"use strict";

(function () {
  const QUERY_ENDPOINT = "/api/query";
  const SUMMARY_ENDPOINT = "/api/summary";
  const SOURCE_ENDPOINT = "/api/source";
  const HEALTH_ENDPOINT = "/health";
  const MAX_QUESTION_LENGTH = 500;
  // Generous: the free hosting instance can take a while to wake up.
  const REQUEST_TIMEOUT_MS = 75000;
  // The only external host ever rendered as a link (besides the fixed repository link).
  const APPROVED_SOURCE_HOST = "www.inegi.org.mx";

  const MODE_LABELS = {
    deterministic: "Exact calculation",
    generated: "Exact calculation · explained by Gemini",
    extractive: "Dataset facts",
    insufficient_data: "Needs more detail",
    unsupported: "Not supported",
  };
  const INTENT_LABELS = {
    count: "Count",
    percentage: "Percentage",
    ranking: "Ranking",
    comparison: "Comparison",
    distribution: "Employment-size distribution",
    examples: "Examples",
    unknown: "Open question",
  };
  const BASIS_TEXT = {
    complete_configured_dataset: "Based on the complete configured dataset (five boroughs, ten categories — not a citywide view).",
    filtered_subset: "Based on a filtered subset of the configured dataset (five boroughs, ten categories — not a citywide view).",
    partial_dataset: "Based on a partial dataset: some borough/category scopes failed to download. See Methodology.",
  };

  const $ = function (id) {
    return document.getElementById(id);
  };
  const form = $("query-form");
  const input = $("question");
  const submitButton = $("submit");
  const results = $("results");
  const placeholder = $("placeholder");
  const loading = $("loading");
  const errorBox = $("error");
  const errorText = $("error-text");
  const retryButton = $("retry");
  const result = $("result");
  const statusLive = $("status-live");
  const serviceStatus = $("service-status");

  let inFlight = false;
  let lastQuestion = "";
  // key -> English label, filled from /api/summary so comparison tables can name categories.
  const categoryLabels = {};

  /* ---------- small DOM helpers (textContent only; no HTML string injection) ---------- */

  function el(tag, options, children) {
    const node = document.createElement(tag);
    const opts = options || {};
    if (opts.className) node.className = opts.className;
    if (opts.text !== undefined) node.textContent = opts.text;
    if (opts.attrs) {
      Object.keys(opts.attrs).forEach(function (name) {
        node.setAttribute(name, opts.attrs[name]);
      });
    }
    (children || []).forEach(function (child) {
      if (child) node.appendChild(child);
    });
    return node;
  }

  function clearChildren(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function formatNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US") : String(value);
  }

  function formatPercent(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) + "%" : "";
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

  function categoryLabel(key) {
    return categoryLabels[key] || String(key);
  }

  function lower(text) {
    return typeof text === "string" && text ? text.charAt(0).toLowerCase() + text.slice(1) : "";
  }

  function joinNames(names, conjunction) {
    if (!names.length) return "";
    if (names.length === 1) return names[0];
    return names.slice(0, -1).join(", ") + " " + conjunction + " " + names[names.length - 1];
  }

  /* ---------- phrases built from the response's applied filters ---------- */

  function boroughPhrase(filters) {
    return filters.boroughs.length ? joinNames(filters.boroughs, "and") : "the five covered boroughs";
  }

  function categoryPhrase(filters, fallback) {
    if (!filters.categories.length) return fallback;
    return joinNames(filters.categories.map(lower), "and");
  }

  function strataPhrase(filters) {
    return filters.strata.length ? " with " + joinNames(filters.strata, "or") + " employed" : "";
  }

  /* ---------- visual building blocks ---------- */

  function barTable(items, options) {
    // A real table is both the accessible data view and the chart: each row carries
    // the label, the exact value, the share, and a decorative bar sized via the CSSOM
    // (inline style attributes are blocked by the Content Security Policy).
    const max = items.reduce(function (acc, item) {
      return Math.max(acc, item.count);
    }, 0);
    const rows = items.map(function (item, index) {
      const fill = el("div", { className: "bar-fill" + (index === 0 && item.count > 0 ? " is-top" : "") });
      fill.style.width = (max > 0 ? (item.count / max) * 100 : 0) + "%";
      const cells = [
        el("th", { attrs: { scope: "row" }, text: item.label }),
        el("td", { className: "bar-cell" }, [el("div", { className: "bar", attrs: { "aria-hidden": "true" } }, [fill])]),
        el("td", { className: "num" }, [
          el("span", { text: formatNumber(item.count) }),
          options.showShare ? el("span", { className: "share-inline", text: " · " + formatPercent(item.share) }) : null,
        ]),
      ];
      if (options.showShare) cells.push(el("td", { className: "num share-col", text: formatPercent(item.share) }));
      return el("tr", {}, cells);
    });
    const head = [
      el("th", { attrs: { scope: "col" }, text: options.labelHeader }),
      el("th", { attrs: { scope: "col" }, text: "" }),
      el("th", { attrs: { scope: "col" }, className: "num", text: "Establishments" }),
    ];
    if (options.showShare) head.push(el("th", { attrs: { scope: "col" }, className: "num share-col", text: "Share" }));
    const table = el("table", { className: "bar-table" }, [
      el("caption", { className: "visually-hidden", text: options.caption }),
      el("thead", {}, [el("tr", {}, head)]),
      el("tbody", {}, rows),
    ]);
    const wrap = el("div", { className: "table-wrap" }, [table]);
    const container = el("div", {}, [wrap]);
    if (options.note) container.appendChild(el("p", { className: "chart-caption", text: options.note }));
    return container;
  }

  function proportionBar(part, whole, partLabel, wholeLabel) {
    const fill = el("div", { className: "bar-fill is-top" });
    fill.style.width = (whole > 0 ? Math.min(100, (part / whole) * 100) : 0) + "%";
    return el("div", { className: "proportion" }, [
      el("div", { className: "bar", attrs: { "aria-hidden": "true" } }, [fill]),
      el("div", { className: "proportion-legend" }, [
        el("span", { text: formatNumber(part) + " " + partLabel }),
        el("span", { text: formatNumber(whole) + " " + wholeLabel }),
      ]),
    ]);
  }

  function comparisonTable(metrics, filters) {
    const boroughs = filters.boroughs.filter(function (name) {
      return metrics[name] && typeof metrics[name] === "object";
    });
    const keys = {};
    boroughs.forEach(function (name) {
      Object.keys(metrics[name].by_category || {}).forEach(function (key) {
        keys[key] = (keys[key] || 0) + metrics[name].by_category[key];
      });
    });
    const ordered = Object.keys(keys).sort(function (a, b) {
      return keys[b] - keys[a] || categoryLabel(a).localeCompare(categoryLabel(b));
    });
    let max = 0;
    boroughs.forEach(function (name) {
      ordered.forEach(function (key) {
        max = Math.max(max, metrics[name].by_category[key] || 0);
      });
    });
    const head = [el("th", { attrs: { scope: "col" }, text: "Category" })].concat(
      boroughs.map(function (name) {
        return el("th", { attrs: { scope: "col" }, className: "num", text: name });
      })
    );
    const rows = ordered.map(function (key) {
      return el(
        "tr",
        {},
        [el("th", { attrs: { scope: "row" }, text: categoryLabel(key) })].concat(
          boroughs.map(function (name) {
            const count = metrics[name].by_category[key] || 0;
            const share = metrics[name].share_pct ? metrics[name].share_pct[key] : undefined;
            const barFill = el("span");
            barFill.style.width = (max > 0 ? (count / max) * 100 : 0) + "%";
            return el("td", { className: "num" }, [
              el("span", { text: formatNumber(count) + (typeof share === "number" ? " (" + formatPercent(share) + ")" : "") }),
              el("div", { className: "cell-bar", attrs: { "aria-hidden": "true" } }, [barFill]),
            ]);
          })
        )
      );
    });
    const foot = el("tr", {}, [el("th", { attrs: { scope: "row" }, text: "All categories" })].concat(
      boroughs.map(function (name) {
        return el("td", { className: "num", text: formatNumber(metrics[name].total) });
      })
    ));
    const table = el("table", { className: "compare-table" }, [
      el("caption", { className: "visually-hidden", text: "Establishments by category for each compared borough" }),
      el("thead", {}, [el("tr", {}, head)]),
      el("tbody", {}, rows),
      el("tfoot", {}, [foot]),
    ]);
    return el("div", {}, [
      el("div", { className: "table-wrap" }, [table]),
      el("p", { className: "chart-caption", text: "Percentages are each borough's share across the covered categories. Bars are scaled to the largest value in the table." }),
    ]);
  }

  function examplesTable(examples) {
    const rows = examples.map(function (item) {
      return el("tr", {}, [
        el("th", { attrs: { scope: "row" }, text: item.name || "" }),
        el("td", { text: item.activity_label || "" }),
        el("td", { text: item.locality || "—" }),
        el("td", { text: item.borough || "" }),
        el("td", { text: item.stratum_label || "" }),
      ]);
    });
    const table = el("table", { className: "data-table" }, [
      el("caption", { className: "visually-hidden", text: "Example establishments" }),
      el("thead", {}, [
        el("tr", {}, [
          el("th", { attrs: { scope: "col" }, text: "Establishment" }),
          el("th", { attrs: { scope: "col" }, text: "DENUE activity class" }),
          el("th", { attrs: { scope: "col" }, text: "Locality" }),
          el("th", { attrs: { scope: "col" }, text: "Borough" }),
          el("th", { attrs: { scope: "col" }, text: "Employment range" }),
        ]),
      ]),
      el("tbody", {}, rows),
    ]);
    return el("div", { className: "table-wrap" }, [table]);
  }

  function factList(answer) {
    const lines = String(answer || "").split("\n").map(function (line) {
      return line.replace(/^•\s*/, "").trim();
    });
    const facts = lines.slice(1).filter(Boolean);
    return el("ul", { className: "fact-list" }, facts.map(function (fact) {
      return el("li", { text: fact });
    }));
  }

  function keyFigure(value, label) {
    return el("div", { className: "key-figure" }, [
      el("span", { className: "value", text: value }),
      el("span", { className: "label", text: label }),
    ]);
  }

  /* ---------- per-intent presentation ---------- */

  function present(data) {
    const metrics = data.metrics || {};
    const filters = {
      boroughs: Array.isArray(data.filters && data.filters.boroughs) ? data.filters.boroughs : [],
      categories: Array.isArray(data.filters && data.filters.categories) ? data.filters.categories : [],
      strata: Array.isArray(data.filters && data.filters.strata) ? data.filters.strata : [],
    };
    const view = { headline: "", figures: [], visual: null, narrative: "", narrativeLabel: "", suggest: false };
    const mode = data.mode;
    const intent = data.intent;

    if (mode === "unsupported") {
      view.headline = "That question isn't supported here";
      view.narrative = data.answer;
      view.suggest = true;
      return view;
    }
    if (mode === "insufficient_data") {
      view.headline = "Not enough detail to compute this";
      view.narrative = data.answer;
      view.suggest = true;
      return view;
    }
    if (mode === "extractive") {
      view.headline = "Related facts from the dataset";
      view.narrativeLabel = "No exact calculation matched this question";
      view.narrative = "These are the closest precomputed facts. Rephrase as a count, ranking, comparison or percentage for an exact result.";
      view.visual = factList(data.answer);
      return view;
    }

    const list = intent === "ranking" ? metrics.ranking : intent === "distribution" ? metrics.by_stratum : null;

    if ((intent === "ranking" || intent === "distribution") && Array.isArray(list) && list.length) {
      const top = list[0];
      const ascending = metrics.order === "lowest to highest";
      const axis = intent === "distribution" ? "stratum" : metrics.axis;
      let labelHeader = "Category";
      if (axis === "borough") {
        labelHeader = "Borough";
        view.headline = top.label + " has the " + (ascending ? "fewest " : "most ") + categoryPhrase(filters, "establishments") + strataPhrase(filters);
      } else if (axis === "stratum") {
        labelHeader = "Employment range";
        view.headline = (ascending ? "Least common employment range among " : "Most common employment range among ") + categoryPhrase(filters, "establishments") + " in " + boroughPhrase(filters) + ": " + top.label;
      } else {
        view.headline = top.label + (ascending ? " is the least common category in " : " is the most common category in ") + boroughPhrase(filters) + strataPhrase(filters);
      }
      view.figures.push(keyFigure(formatNumber(top.count), top.label + (typeof top.share === "number" || typeof top.share_pct === "number" ? " · " + formatPercent(top.share_pct) + " of this selection" : "")));
      view.figures.push(keyFigure(formatNumber(metrics.total), "establishments in this selection"));
      view.visual = barTable(
        list.map(function (item) {
          return { label: item.label, count: item.count, share: item.share_pct };
        }),
        {
          labelHeader: labelHeader,
          showShare: true,
          caption: "Establishments by " + labelHeader.toLowerCase() + ", " + (ascending ? "lowest to highest" : "highest to lowest"),
          note: "Sorted " + (ascending ? "lowest to highest" : "highest to lowest") + ". Shares are relative to the " + formatNumber(metrics.total) + " establishments in this selection.",
        }
      );
      return view;
    }

    if (intent === "count" && typeof metrics.count === "number") {
      view.headline = formatNumber(metrics.count) + " " + categoryPhrase(filters, "establishments") + " in " + boroughPhrase(filters) + strataPhrase(filters);
      const wholeDataset = metrics.count === metrics.dataset_total;
      view.figures.push(keyFigure(formatNumber(metrics.count), wholeDataset ? "establishments in the whole configured dataset" : "establishments matching the question"));
      if (!wholeDataset && typeof metrics.dataset_total === "number" && metrics.dataset_total > 0) {
        const share = (metrics.count / metrics.dataset_total) * 100;
        view.figures.push(keyFigure(formatPercent(share), "of the " + formatNumber(metrics.dataset_total) + " establishments in the dataset"));
        view.visual = proportionBar(metrics.count, metrics.dataset_total, "matching", "in the whole dataset");
      }
      return view;
    }

    if (intent === "percentage" && typeof metrics.percentage === "number") {
      const statement = typeof data.answer === "string" && mode === "deterministic" ? data.answer.split(":")[0] : "";
      view.headline = statement || formatPercent(metrics.percentage) + " of the selection";
      view.figures.push(keyFigure(formatPercent(metrics.percentage), formatNumber(metrics.part) + " of " + formatNumber(metrics.whole) + " establishments"));
      view.visual = proportionBar(metrics.part, metrics.whole, "matching", "in the comparison base");
      return view;
    }

    if (intent === "comparison") {
      if (metrics.by_category && typeof metrics.total === "number") {
        const items = Object.keys(metrics.by_category)
          .map(function (key) {
            const count = metrics.by_category[key];
            return { label: categoryLabel(key), count: count, share: metrics.total > 0 ? (count / metrics.total) * 100 : 0 };
          })
          .sort(function (a, b) {
            return b.count - a.count;
          });
        view.headline = joinNames(items.map(function (item) { return item.label; }), "vs") + " in " + boroughPhrase(filters) + strataPhrase(filters);
        view.figures.push(keyFigure(formatNumber(metrics.total), "establishments across these categories"));
        view.visual = barTable(items, {
          labelHeader: "Category",
          showShare: true,
          caption: "Establishments by compared category",
          note: "Shares are relative to the " + formatNumber(metrics.total) + " establishments in the compared categories.",
        });
        return view;
      }
      const compared = filters.boroughs.filter(function (name) {
        return metrics[name] && typeof metrics[name] === "object";
      });
      if (compared.length >= 2) {
        view.headline = joinNames(compared, "vs") + (filters.categories.length ? " — " + categoryPhrase(filters, "") : "") + strataPhrase(filters);
        compared.forEach(function (name) {
          view.figures.push(keyFigure(formatNumber(metrics[name].total), "establishments in " + name));
        });
        view.visual = comparisonTable(metrics, filters);
        return view;
      }
    }

    if (intent === "examples" && Array.isArray(metrics.examples)) {
      view.headline = formatNumber(metrics.sample_size) + " of " + formatNumber(metrics.matching) + " " + categoryPhrase(filters, "establishments") + " in " + boroughPhrase(filters) + strataPhrase(filters);
      view.figures.push(keyFigure(formatNumber(metrics.matching), "matching establishments (sample sorted by name)"));
      view.visual = examplesTable(metrics.examples);
      return view;
    }

    // Unknown shape: fall back to the server's own statement.
    view.headline = INTENT_LABELS[intent] || "Result";
    view.narrative = data.answer;
    return view;
  }

  /* ---------- rendering ---------- */

  function renderFilters(filters) {
    const list = $("filters");
    clearChildren(list);
    const tags = []
      .concat(filters.boroughs.map(function (name) { return "Borough: " + name; }))
      .concat(filters.categories.map(function (name) { return "Category: " + name; }))
      .concat(filters.strata.map(function (name) { return "Employment: " + name; }));
    if (!tags.length) tags.push("All five boroughs", "All ten categories");
    tags.forEach(function (text) {
      list.appendChild(el("li", { className: "tag", text: text }));
    });
    $("filters-block").hidden = false;
  }

  function renderEvidence(evidence) {
    const list = $("evidence");
    clearChildren(list);
    evidence.forEach(function (item) {
      if (!item || typeof item !== "object") return;
      list.appendChild(
        el("li", {}, [
          el("span", { className: "evidence-kind", text: item.kind === "aggregate" ? "Calculated aggregate" : "Sample establishment record" }),
          el("span", { text: typeof item.text === "string" ? item.text : "" }),
        ])
      );
    });
    $("evidence-details").hidden = evidence.length === 0;
    $("evidence-details").open = false;
  }

  function renderTechnical(data) {
    const list = $("tech-list");
    clearChildren(list);
    const rows = [
      ["Response mode", MODE_LABELS[data.mode] || String(data.mode)],
      ["Detected intent", INTENT_LABELS[data.intent] || String(data.intent)],
      ["Dataset basis", data.scope && data.scope.basis ? String(data.scope.basis).replace(/_/g, " ") : ""],
      ["Data retrieved", data.scope && typeof data.scope.retrieved_at === "string" ? data.scope.retrieved_at.slice(0, 10) : ""],
    ];
    const evidence = Array.isArray(data.evidence) ? data.evidence : [];
    if (evidence.length) {
      rows.push(["Evidence similarity", evidence.map(function (item) { return typeof item.score === "number" ? item.score.toFixed(2) : "?"; }).join(", ")]);
    }
    if (data.mode === "deterministic" && typeof data.answer === "string") rows.push(["Exact statement", data.answer]);
    rows.forEach(function (row) {
      if (!row[1]) return;
      list.appendChild(el("dt", { text: row[0] }));
      list.appendChild(el("dd", { text: row[1] }));
    });
    $("tech-details").open = false;
  }

  function renderSuggestions(container) {
    clearChildren(container);
    document.querySelectorAll(".suggestions .chip[data-question]").forEach(function (chip, index) {
      if (index > 2) return;
      const button = el("button", { className: "chip", text: chip.textContent, attrs: { type: "button" } });
      button.addEventListener("click", function () {
        input.value = chip.dataset.question || "";
        submitQuestion(input.value);
      });
      container.appendChild(button);
    });
  }

  function renderResponse(data) {
    const view = present(data);
    const kicker = [INTENT_LABELS[data.intent] || "Result", MODE_LABELS[data.mode] || ""].filter(Boolean).join(" · ");
    $("result-kicker").textContent = kicker;
    $("result-headline").textContent = view.headline;

    const figures = $("key-figures");
    clearChildren(figures);
    view.figures.forEach(function (figure) { figures.appendChild(figure); });
    figures.hidden = view.figures.length === 0;

    const narrativeBlock = $("narrative-block");
    let narrative = view.narrative;
    let narrativeLabel = view.narrativeLabel;
    if (data.mode === "generated" && typeof data.answer === "string") {
      narrative = data.answer;
      narrativeLabel = "Explanation · AI-assisted, checked against the calculated values";
    }
    $("narrative").textContent = narrative || "";
    $("narrative-label").textContent = narrativeLabel || "";
    $("narrative-label").hidden = !narrativeLabel;
    narrativeBlock.hidden = !narrative;

    const visual = $("visual");
    clearChildren(visual);
    if (view.visual) visual.appendChild(view.visual);

    const suggested = $("suggested-actions");
    suggested.hidden = !view.suggest;
    if (view.suggest) renderSuggestions($("suggested-buttons"));

    const filters = {
      boroughs: Array.isArray(data.filters && data.filters.boroughs) ? data.filters.boroughs : [],
      categories: Array.isArray(data.filters && data.filters.categories) ? data.filters.categories : [],
      strata: Array.isArray(data.filters && data.filters.strata) ? data.filters.strata : [],
    };
    if (data.mode === "unsupported") {
      $("filters-block").hidden = true;
    } else {
      renderFilters(filters);
    }

    const scopeNote = $("scope-note");
    const basis = data.scope && BASIS_TEXT[data.scope.basis] ? data.scope.basis : null;
    scopeNote.textContent = basis && data.mode !== "unsupported" ? BASIS_TEXT[basis] : "";
    scopeNote.dataset.partial = basis === "partial_dataset" ? "true" : "false";

    renderEvidence(Array.isArray(data.evidence) ? data.evidence : []);
    $("methodology").textContent = typeof data.methodology === "string" ? data.methodology : "";
    $("result-attribution").textContent = typeof data.attribution === "string" ? data.attribution : "";
    if (data.source && isApprovedUrl(data.source.url)) $("source-link").href = data.source.url;
    $("method-details").open = false;
    renderTechnical(data);

    result.hidden = false;
    statusLive.textContent = "Analysis ready: " + view.headline;
  }

  /* ---------- request lifecycle ---------- */

  function showState(state) {
    placeholder.hidden = state !== "idle";
    loading.hidden = state !== "loading";
    errorBox.hidden = state !== "error";
    if (state === "loading" || state === "error") result.hidden = true;
    results.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  }

  function setBusy(busy) {
    inFlight = busy;
    input.disabled = busy;
    submitButton.disabled = busy;
    submitButton.textContent = busy ? "Analyzing…" : "Analyze";
  }

  function showError(message) {
    errorText.textContent = message;
    showState("error");
    statusLive.textContent = "Error: " + message;
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
    const question = String(rawQuestion || "").replace(/\s+/g, " ").trim();
    if (!question) {
      showError("Please enter a question first.");
      input.focus();
      return;
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      showError("Questions are limited to " + MAX_QUESTION_LENGTH + " characters.");
      return;
    }
    if (inFlight) return;

    lastQuestion = question;
    setBusy(true);
    showState("loading");
    statusLive.textContent = "Analyzing the selected dataset…";
    const controller = new AbortController();
    const timer = setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(QUERY_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ question: question }),
        signal: controller.signal,
      });
      if (!response.ok) {
        showError(await readErrorMessage(response));
        return;
      }
      renderResponse(await response.json());
      showState("result");
    } catch (error) {
      if (error && error.name === "AbortError") {
        showError("The request timed out. The free hosting instance may have been waking up — please retry.");
      } else {
        showError("Could not reach the service. Check your connection and retry.");
      }
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  /* ---------- dataset overview ---------- */

  function fillTable(tableId, rows) {
    const body = $(tableId).querySelector("tbody");
    clearChildren(body);
    rows.forEach(function (row) { body.appendChild(row); });
  }

  async function loadSummary() {
    try {
      const response = await fetch(SUMMARY_ENDPOINT, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("summary unavailable");
      const s = await response.json();
      const total = typeof s.total_establishments === "number" ? s.total_establishments : 0;
      const boroughs = Array.isArray(s.boroughs) ? s.boroughs : [];
      const categories = Array.isArray(s.categories) ? s.categories : [];
      categories.forEach(function (c) {
        if (c && typeof c.key === "string" && typeof c.label_en === "string") categoryLabels[c.key] = c.label_en;
      });
      $("stat-total").textContent = formatNumber(total);
      $("stat-boroughs").textContent = formatNumber(boroughs.length);
      $("stat-categories").textContent = formatNumber(categories.length);
      // The scope shown under the question box comes from the dataset itself, so a
      // reconfigured ingestion never leaves the page describing the wrong coverage.
      $("scope-boroughs").textContent = joinNames(boroughs.map(function (b) { return b.name; }), "and");
      $("scope-categories").textContent = joinNames(categories.map(function (c) { return lower(c.label_en); }), "and");

      const retrieved = typeof s.retrieved_at === "string" ? s.retrieved_at.slice(0, 10) : "unknown date";
      const coverage = s.complete
        ? "complete for the configured scope (" + boroughs.length + " boroughs, " + categories.length + " categories) — not a citywide view"
        : "partial: some scopes failed to download; see Scope and limitations";
      $("dataset-meta").textContent = "INEGI DENUE data retrieved " + retrieved + " · Coverage " + coverage + ".";

      fillTable("borough-table", boroughs.map(function (b) {
        return el("tr", {}, [
          el("th", { attrs: { scope: "row" }, text: b.name }),
          el("td", { className: "num", text: formatNumber(b.count) }),
          el("td", { className: "num", text: total > 0 ? formatPercent((b.count / total) * 100) : "" }),
        ]);
      }));
      fillTable("category-table", categories.map(function (c) {
        return el("tr", {}, [
          el("th", { attrs: { scope: "row" } }, [el("span", { text: c.label_en }), el("span", { className: "sublabel", text: c.label_es })]),
          el("td", { className: "num", text: formatNumber(c.count) }),
          el("td", { className: "num", text: total > 0 ? formatPercent((c.count / total) * 100) : "" }),
        ]);
      }));
      if (typeof s.attribution === "string") $("footer-attribution").textContent = s.attribution;
    } catch (_error) {
      $("dataset-meta").textContent = "Dataset information is not available right now.";
      $("scope-boroughs").textContent = "the configured Mexico City boroughs";
      $("scope-categories").textContent = "the configured retail categories";
    }
  }

  async function loadSource() {
    try {
      const response = await fetch(SOURCE_ENDPOINT, { headers: { Accept: "application/json" } });
      if (!response.ok) return;
      const source = await response.json();
      $("completeness-note").textContent = typeof source.completeness_note === "string" ? source.completeness_note : "";
    } catch (_error) {
      // Optional detail; the overview already states completeness.
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
        serviceStatus.textContent = "AI explanations on";
        serviceStatus.dataset.state = "ok";
      } else if (payload.generation_configured) {
        serviceStatus.textContent = "AI explanations paused · exact answers only";
        serviceStatus.dataset.state = "degraded";
      } else {
        serviceStatus.textContent = "Exact answers · no language model";
        serviceStatus.dataset.state = "ok";
      }
    } catch (_error) {
      serviceStatus.textContent = "Service status unknown";
      serviceStatus.dataset.state = "unavailable";
    }
  }

  /* ---------- wiring ---------- */

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuestion(input.value);
  });

  retryButton.addEventListener("click", function () {
    if (lastQuestion) {
      input.value = lastQuestion;
      submitQuestion(lastQuestion);
    } else {
      input.focus();
    }
  });

  document.querySelectorAll(".suggestions .chip[data-question]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      input.value = chip.dataset.question || "";
      submitQuestion(input.value);
    });
  });

  showState("idle");
  loadSummary();
  loadSource();
  loadServiceStatus();
})();
