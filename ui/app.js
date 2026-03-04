const form = document.getElementById("query-form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit-button");
const statusLine = document.getElementById("status-line");
const routingLine = document.getElementById("routing-line");
const composerPreview = document.getElementById("composer-preview");
const answerCard = document.getElementById("answer-card");
const answerContent = document.getElementById("answer-content");
const confidenceCard = document.getElementById("confidence-card");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceNote = document.getElementById("confidence-note");
const citationsList = document.getElementById("citations-list");
const matchesList = document.getElementById("matches-list");
const metricsGrid = document.getElementById("metrics-grid");
const evidenceCount = document.getElementById("evidence-count");
const inlineProgress = document.getElementById("inline-progress");
const topKInput = document.getElementById("top_k");
const topKValue = document.getElementById("top_k_value");
const citationTemplate = document.getElementById("citation-template");
const matchTemplate = document.getElementById("match-template");
const progressChips = Array.from(document.querySelectorAll(".progress-chip"));
const exampleButtons = Array.from(document.querySelectorAll(".example-chip"));

let stageTimers = [];
let flashTimerId = 0;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("error-text", isError);
}

function formatDistance(value) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return value.toFixed(4);
}

function estimateConfidence(distance) {
  if (typeof distance !== "number") {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round((1 - Math.min(distance, 1)) * 100)));
}

function percentFromScore(score) {
  if (typeof score !== "number") {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round(score * 100)));
}

function formatPercent(score, fallbackDistance = null) {
  const scorePercent = percentFromScore(score);
  if (scorePercent !== null) {
    return `${scorePercent}%`;
  }
  const distancePercent = estimateConfidence(fallbackDistance);
  return distancePercent === null ? "n/a" : `${distancePercent}%`;
}

function getEvidenceItems(data) {
  if (Array.isArray(data.evidence)) {
    return data.evidence;
  }
  if (Array.isArray(data.matches)) {
    return data.matches;
  }
  return [];
}

function getCitationItems(data) {
  if (Array.isArray(data.citations) && data.citations.length && typeof data.citations[0] === "object") {
    return data.citations;
  }
  return [];
}

function getAnswerSegments(data) {
  if (Array.isArray(data.answer_segments) && data.answer_segments.length) {
    return data.answer_segments;
  }
  return [];
}

function resetStageTimers() {
  stageTimers.forEach((timerId) => window.clearTimeout(timerId));
  stageTimers = [];
}

function setProgressState(stage, active) {
  const target = progressChips.find((chip) => chip.dataset.stage === stage);
  if (!target) {
    return;
  }
  target.classList.toggle("is-active", Boolean(active));
}

function hideProgress() {
  resetStageTimers();
  progressChips.forEach((chip) => chip.classList.remove("is-active"));
  inlineProgress.hidden = true;
}

function startLoadingStages(rerankerEnabled) {
  resetStageTimers();
  inlineProgress.hidden = false;
  progressChips.forEach((chip) => chip.classList.remove("is-active"));
  setProgressState("searching", true);
  setStatus("Searching regulatory corpus...");

  stageTimers.push(
    window.setTimeout(() => {
      setProgressState("searching", false);
      setProgressState("retrieving", true);
      setStatus(rerankerEnabled ? "Retrieving documents and reranking..." : "Retrieving documents...");
    }, 320),
  );

  stageTimers.push(
    window.setTimeout(() => {
      setProgressState("retrieving", false);
      setProgressState("generating", true);
      setStatus("Generating answer...");
    }, 760),
  );
}

function updateComposerPreview() {
  const text = questionInput.value.trim();
  composerPreview.textContent = text
    ? `Current question: ${text}`
    : "Results will appear below with grounded sources.";
}

function resetOutputForNewRequest() {
  answerCard.classList.remove("error");
  answerContent.classList.add("empty-text");
  answerContent.textContent = "Working on your query...";
  confidenceCard.classList.add("is-empty");
  confidenceValue.textContent = "n/a";
  confidenceFill.style.width = "0%";
  confidenceNote.textContent = "Derived from citation scores.";
  routingLine.textContent = "Processing query...";
  citationsList.className = "sources-list";
  citationsList.textContent = "Preparing sources...";
  matchesList.className = "evidence-list";
  matchesList.textContent = "Preparing retrieved chunks...";
  evidenceCount.textContent = "0";
  metricsGrid.innerHTML = `
    <div class="metric-tile"><span class="metric-label">Top K</span><strong>...</strong></div>
    <div class="metric-tile"><span class="metric-label">Reranker</span><strong>...</strong></div>
    <div class="metric-tile"><span class="metric-label">Embedding</span><strong>...</strong></div>
    <div class="metric-tile"><span class="metric-label">Docs Retrieved</span><strong>...</strong></div>
  `;
}

function renderConfidence(data) {
  const citations = getCitationItems(data);
  const confidenceScores = citations
    .map((entry) => percentFromScore(entry.score))
    .filter((value) => typeof value === "number");

  if (!confidenceScores.length) {
    confidenceCard.classList.add("is-empty");
    confidenceValue.textContent = "n/a";
    confidenceFill.style.width = "0%";
    confidenceNote.textContent = "Derived from citation scores. No grounded citations were available.";
    return;
  }

  const averageConfidence = Math.round(
    confidenceScores.reduce((sum, value) => sum + value, 0) / confidenceScores.length,
  );
  confidenceCard.classList.remove("is-empty");
  confidenceValue.textContent = `${averageConfidence}%`;
  confidenceFill.style.width = `${averageConfidence}%`;
  confidenceNote.textContent =
    `Derived from ${confidenceScores.length} citation source${confidenceScores.length === 1 ? "" : "s"}.`;
}

function highlightTerms(text, query) {
  const words = String(query)
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9_-]*/g) || [];
  const terms = [...new Set(words.filter((word) => word.length > 3))].slice(0, 6);
  let highlighted = escapeHtml(String(text || ""));

  terms.forEach((term) => {
    const pattern = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig");
    highlighted = highlighted.replace(pattern, '<mark class="chunk-highlight">$1</mark>');
  });
  return highlighted;
}

function renderCitationAnchors(text, citations) {
  const citationMap = new Map(
    citations.map((entry) => [String(entry.citation_number), entry]),
  );

  return escapeHtml(text).replace(/\[(\d+)\]/g, (match, number) => {
    const citation = citationMap.get(number);
    if (!citation) {
      return match;
    }
    const tooltipLines = [
      `<strong>${escapeHtml(citation.doc_title || "Unknown document")}</strong>`,
      `<span>Page ${escapeHtml(citation.page ?? "n/a")}</span>`,
      `<span>${escapeHtml(citation.snippet || "No preview available.")}</span>`,
    ].join("");
    return `<button type="button" class="citation-anchor" data-citation-number="${escapeHtml(number)}" data-chunk-id="${escapeHtml(citation.chunk_id || "")}" aria-label="Open source ${escapeHtml(number)}">[${escapeHtml(number)}]<span class="citation-tooltip">${tooltipLines}</span></button>`;
  });
}

function flashEvidenceCard(target) {
  if (flashTimerId) {
    window.clearTimeout(flashTimerId);
    flashTimerId = 0;
  }
  target.classList.remove("evidence-flash");
  void target.offsetWidth;
  target.classList.add("evidence-flash");
  flashTimerId = window.setTimeout(() => {
    target.classList.remove("evidence-flash");
    flashTimerId = 0;
  }, 1800);
}

function scrollToEvidenceChunk(chunkId) {
  if (!chunkId) {
    return;
  }
  const target = document.getElementById(`evidence-${chunkId}`);
  if (!target) {
    return;
  }
  target.open = true;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  flashEvidenceCard(target);
}

function renderAnswer(data) {
  answerCard.classList.remove("error");
  answerContent.classList.remove("empty-text");

  if (data.error) {
    answerCard.classList.add("error");
    answerContent.textContent = data.error;
    return;
  }

  const payload = data.answer_json || {};
  const citations = getCitationItems(data);
  const answerSegments = getAnswerSegments(data);
  const answerMarkdown = String(data.answer_markdown || payload.answer || "").trim();
  const blocks = [];

  if (answerSegments.length) {
    const allBulletLike = answerSegments.every((segment) => String(segment.text || "").trim().startsWith("- "));
    const segmentsMarkup = answerSegments
      .map((segment) => {
        const rawText = String(segment.text || "").trim();
        const normalizedText = allBulletLike && rawText.startsWith("- ")
          ? rawText.slice(2).trim()
          : rawText;
        const text = escapeHtml(normalizedText);
        if (!text) {
          return "";
        }
        const markers = (Array.isArray(segment.citation_numbers) ? segment.citation_numbers : [])
          .map((number) => {
            const citation = citations.find((entry) => Number(entry.citation_number) === Number(number));
            if (!citation) {
              return "";
            }
            const tooltipLines = [
              `<strong>${escapeHtml(citation.doc_title || "Unknown document")}</strong>`,
              `<span>Page ${escapeHtml(citation.page ?? "n/a")}</span>`,
              `<span>${escapeHtml(citation.snippet || "No preview available.")}</span>`,
            ].join("");
            return `<button type="button" class="citation-anchor" data-citation-number="${escapeHtml(number)}" data-chunk-id="${escapeHtml(citation.chunk_id || "")}" aria-label="Open source ${escapeHtml(number)}">[${escapeHtml(number)}]<span class="citation-tooltip">${tooltipLines}</span></button>`;
          })
          .filter(Boolean)
          .join(" ");
        const suffix = markers ? ` ${markers}` : "";
        return allBulletLike
          ? `<li class="answer-item">${text}${suffix}</li>`
          : `<p class="answer-copy">${text}${suffix}</p>`;
      })
      .filter(Boolean)
      .join("");
    if (segmentsMarkup) {
      blocks.push(allBulletLike ? `<ul class="answer-list">${segmentsMarkup}</ul>` : segmentsMarkup);
    }
  } else if (answerMarkdown) {
    const paragraphs = answerMarkdown
      .split(/\n{2,}/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean)
      .map((paragraph) => `<p class="answer-copy">${renderCitationAnchors(paragraph, citations)}</p>`)
      .join("");
    if (paragraphs) {
      blocks.push(paragraphs);
    }
  }

  if (Array.isArray(payload.notes) && payload.notes.length) {
    const notesMarkup = payload.notes
      .map((note) => `<li>${escapeHtml(note)}</li>`)
      .join("");
    blocks.push(`
      <div class="answer-notes">
        <p class="answer-notes-title">Grounding notes</p>
        <ul>${notesMarkup}</ul>
      </div>
    `);
  }

  answerContent.innerHTML = blocks.join("") || "No answer returned.";
}

function renderCitations(data) {
  clearNode(citationsList);
  const citations = getCitationItems(data);

  if (!citations.length) {
    citationsList.className = "sources-list empty-text";
    citationsList.textContent = "No sources returned.";
    return;
  }

  citationsList.className = "sources-list";

  citations.forEach((entry) => {
    const fragment = citationTemplate.content.cloneNode(true);
    const scoreText = formatPercent(entry.score);
    const sourceCard = fragment.querySelector(".source-card");
    const jumpButton = fragment.querySelector(".source-jump-button");
    sourceCard.dataset.chunkId = entry.chunk_id || "";
    jumpButton.dataset.chunkId = entry.chunk_id || "";
    jumpButton.disabled = !entry.chunk_id;
    fragment.querySelector(".source-title").textContent =
      `[${entry.citation_number}] ${entry.doc_title || "Unknown document"}`;
    fragment.querySelector(".source-score").textContent = scoreText;
    fragment.querySelector(".source-meta").textContent =
      `Page ${entry.page ?? "n/a"} | confidence ${scoreText}`;
    fragment.querySelector(".source-body").innerHTML = highlightTerms(
      entry.quote || entry.snippet || "No citation quote available.",
      data.question,
    );
    citationsList.appendChild(fragment);
  });
}

function renderMetrics(data) {
  const config = data.request_config || {};
  const evidence = getEvidenceItems(data);
  const uniqueDocs = new Set(evidence.map((match) => String(match.doc_title || match.doc || ""))).size;

  metricsGrid.innerHTML = `
    <div class="metric-tile">
      <span class="metric-label">Top K</span>
      <strong>${escapeHtml(config.top_k ?? "n/a")}</strong>
    </div>
    <div class="metric-tile">
      <span class="metric-label">Reranker</span>
      <strong>${config.reranker_enabled ? "Enabled" : "Disabled"}</strong>
    </div>
    <div class="metric-tile">
      <span class="metric-label">Embedding</span>
      <strong>${escapeHtml(data.embedding_model || "n/a")}</strong>
    </div>
    <div class="metric-tile">
      <span class="metric-label">Docs Retrieved</span>
      <strong>${uniqueDocs}</strong>
    </div>
  `;
}

function renderMatches(data) {
  clearNode(matchesList);
  const evidence = getEvidenceItems(data);
  evidenceCount.textContent = String(evidence.length);

  if (!evidence.length) {
    matchesList.className = "evidence-list empty-text";
    matchesList.textContent = "No retrieved chunks returned.";
    return;
  }

  matchesList.className = "evidence-list";

  evidence.forEach((match) => {
    const fragment = matchTemplate.content.cloneNode(true);
    const evidenceItem = fragment.querySelector(".evidence-item");
    const scoreText = formatPercent(match.score, match.distance);
    evidenceItem.id = `evidence-${match.chunk_id}`;
    evidenceItem.dataset.chunkId = match.chunk_id;
    fragment.querySelector(".chunk-title").textContent = match.doc_title || match.doc || "Unknown document";
    fragment.querySelector(".chunk-score").textContent = `score ${scoreText}`;
    fragment.querySelector(".chunk-meta").textContent =
      `Page ${match.page ?? "n/a"} | chunk ${match.chunk ?? "n/a"} | distance ${formatDistance(match.distance)}`;
    fragment.querySelector(".chunk-body").innerHTML = highlightTerms(
      match.text || match.snippet || "No snippet available.",
      data.question,
    );
    fragment.querySelector(".chunk-path").textContent = match.source_path || "No source path available.";
    matchesList.appendChild(fragment);
  });
}

async function runQuery(payload) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Query failed.");
  }
  return data;
}

function syncTopKValue() {
  topKValue.textContent = topKInput.value;
}

topKInput.addEventListener("input", syncTopKValue);
syncTopKValue();

questionInput.addEventListener("input", updateComposerPreview);
updateComposerPreview();

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question || "";
    updateComposerPreview();
    questionInput.focus();
  });
});

answerContent.addEventListener("click", (event) => {
  const trigger = event.target.closest(".citation-anchor");
  if (!trigger) {
    return;
  }
  event.preventDefault();
  scrollToEvidenceChunk(trigger.dataset.chunkId || "");
});

citationsList.addEventListener("toggle", (event) => {
  const sourceCard = event.target.closest(".source-card");
  if (!sourceCard || !sourceCard.open) {
    return;
  }
  const chunkId = sourceCard.dataset.chunkId || "";
  if (chunkId) {
    scrollToEvidenceChunk(chunkId);
  }
});

citationsList.addEventListener("click", (event) => {
  const trigger = event.target.closest(".source-jump-button");
  if (!trigger) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  scrollToEvidenceChunk(trigger.dataset.chunkId || "");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const currentQuestion = questionInput.value.trim();
  if (!currentQuestion) {
    questionInput.focus();
    questionInput.reportValidity();
    return;
  }

  const payload = {
    question: currentQuestion,
    topic: document.getElementById("topic").value,
    doc_family: document.getElementById("doc_family").value,
    top_k: Number(topKInput.value),
    min_citations: Number(document.getElementById("min_citations").value),
    reranker_enabled: document.getElementById("reranker_enabled").checked,
  };

  submitButton.disabled = true;
  composerPreview.textContent = `Running: ${currentQuestion}`;
  resetOutputForNewRequest();
  startLoadingStages(payload.reranker_enabled);

  try {
    const data = await runQuery(payload);
    hideProgress();
    routingLine.textContent =
      `Resolved routing: topic = ${data.effective_topic || "auto"} | doc family = ${data.effective_doc_family || "auto"}`;
    renderAnswer(data);
    renderConfidence(data);
    renderCitations(data);
    renderMetrics(data);
    renderMatches(data);

    if (Array.isArray(data.validation_reasons) && data.validation_reasons.length) {
      setStatus(`Completed with guardrail notes: ${data.validation_reasons.join(", ")}`);
    } else {
      setStatus("Completed.");
    }
  } catch (error) {
    hideProgress();
    answerCard.classList.add("error");
    answerContent.classList.remove("empty-text");
    answerContent.textContent = error.message;
    setStatus(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
});
