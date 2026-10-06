/* Browser interface for the document pipeline. */

const API = "/api";

const $ = (id) => document.getElementById(id);

let selectedFile = null;

/* ---------- helpers ---------- */

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

async function request(path, options = {}) {
  const response = await fetch(API + path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || body.error || `Request failed (${response.status})`);
  }
  return body;
}

/* A long running form: shows elapsed time and can be stopped. */
function createRunner(name) {
  const submit = $(`${name}-submit`);
  const stop = $(`${name}-stop`);
  const spinner = $(`${name}-spinner`);
  const elapsed = $(`${name}-elapsed`);
  let controller = null;
  let ticker = null;

  stop.addEventListener("click", () => {
    if (controller) {
      stop.disabled = true;
      controller.abort();
    }
  });

  function show(seconds) {
    elapsed.textContent =
      seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }

  return {
    begin() {
      controller = new AbortController();
      const startedAt = Date.now();

      submit.disabled = true;
      stop.disabled = false;
      stop.hidden = false;
      spinner.hidden = false;
      elapsed.hidden = false;
      show(0);
      ticker = setInterval(() => show(Math.round((Date.now() - startedAt) / 1000)), 1000);

      return controller.signal;
    },
    end() {
      clearInterval(ticker);
      ticker = null;
      controller = null;
      submit.disabled = false;
      stop.hidden = true;
      spinner.hidden = true;
      elapsed.hidden = true;
    },
  };
}

function wasStopped(error) {
  return error.name === "AbortError";
}

const stoppedNotice = '<div class="notice warn">Stopped before a response arrived.</div>';

function errorNotice(message) {
  return `<div class="notice error">${escapeHtml(message)}</div>`;
}

function textBlock(label, text, extra = "") {
  if (!text) return "";
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return `
    <div class="block-label"><span>${escapeHtml(label)}</span>
      <span>${words} words${extra ? " &middot; " + escapeHtml(extra) : ""}</span></div>
    <pre class="text-block">${escapeHtml(text)}</pre>`;
}

function languagePills(languages) {
  if (!languages || !languages.length) {
    return '<span class="pill">No language identified</span>';
  }
  return languages
    .map((item) => `<span class="pill ok">${escapeHtml(item.name)} ${Math.round(item.share * 100)}%</span>`)
    .join("");
}

/* ---------- tabs ---------- */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((other) => other.classList.remove("is-active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("is-active"));
    tab.classList.add("is-active");
    $(`panel-${tab.dataset.panel}`).classList.add("is-active");
    if (tab.dataset.panel === "library" || tab.dataset.panel === "ask") loadLibrary();
  });
});

/* ---------- status ---------- */

async function loadStatus() {
  const container = $("status");
  try {
    const health = await request("/health");
    const parts = [
      `<span class="pill ${health.tesseract_available ? "ok" : "bad"}">Recognition ${
        health.tesseract_available ? "ready" : "unavailable"}</span>`,
      `<span class="pill ${health.model_provider_configured ? "ok" : "warn"}">Consciousnx ${
        health.model_provider_configured ? "configured" : "not configured"}</span>`,
      `<span class="pill ${health.embedding_model_ready ? "ok" : "warn"}">Search ${
        health.embedding_model_ready ? "ready" : "loading"}</span>`,
      `<span class="pill">${health.document_count} documents &middot; ${health.chunk_count} chunks</span>`,
    ];
    container.innerHTML = parts.join("");
  } catch (error) {
    container.innerHTML = `<span class="pill bad">Backend unreachable</span>`;
  }
}

/* Used when the language list cannot be fetched. */
const FALLBACK_TARGETS = [
  "English", "Hindi", "Bengali", "Punjabi", "Gujarati",
  "Odia", "Tamil", "Telugu", "Kannada", "Malayalam",
];

function fillTargetLanguages(names) {
  const options = names
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");

  ["target-language", "translate-target"].forEach((id) => {
    const select = $(id);
    if (!select) return;
    const previous = select.value;
    select.innerHTML = options;
    select.value = names.includes(previous) ? previous : "English";
  });
}

async function loadLanguages() {
  const select = $("language-select");
  try {
    const languages = await request("/languages");
    select.innerHTML = languages
      .map((language) => {
        const label = language.installed ? language.name : `${language.name} (not installed)`;
        const disabled = language.installed ? "" : " disabled";
        return `<option value="${language.code}"${disabled}>${escapeHtml(label)}</option>`;
      })
      .join("");
    fillTargetLanguages(languages.map((language) => language.name));
  } catch (error) {
    select.innerHTML = "";
    fillTargetLanguages(FALLBACK_TARGETS);
  }
}

/* ---------- file selection ---------- */

const dropzone = $("dropzone");
const fileInput = $("file-input");

function setFile(file) {
  selectedFile = file;
  $("file-name").textContent = file ? `${file.name} (${Math.round(file.size / 1024)} KB)` : "";
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));

["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-hot");
  })
);

["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-hot");
  })
);

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});

/* ---------- process ---------- */

$("process-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const output = $("process-result");

  if (!selectedFile) {
    output.innerHTML = errorNotice("Choose a file before processing.");
    return;
  }

  const chosen = Array.from($("language-select").selectedOptions).map((option) => option.value);
  const form = new FormData();
  form.append("file", selectedFile);
  if (chosen.length) form.append("language", chosen.join("+"));
  form.append("correct", $("opt-correct").checked);
  form.append("translate", $("opt-translate").checked);
  form.append("target_language", $("target-language").value || "English");
  form.append("index", $("opt-index").checked);
  form.append("strategy", $("strategy").value);

  const signal = processRunner.begin();
  output.innerHTML = "";

  try {
    const document_ = await request("/documents", { method: "POST", body: form, signal });
    output.innerHTML = renderDocument(document_, "Result");
    loadStatus();
    loadLibrary();
  } catch (error) {
    output.innerHTML = wasStopped(error) ? stoppedNotice : errorNotice(error.message);
  } finally {
    processRunner.end();
  }
});

function renderDocument(document_, heading) {
  const meta = [
    `<span class="pill">Document ${document_.id}</span>`,
    `<span class="pill">${escapeHtml(document_.source_kind)}</span>`,
    languagePills(document_.detected_languages),
    `<span class="pill">${document_.chunk_count} chunks indexed</span>`,
    `<span class="pill">${document_.processing_time_ms} ms</span>`,
  ].join("");

  return `
    <div class="result">
      <h3>${escapeHtml(heading)}: ${escapeHtml(document_.filename)}</h3>
      <div class="meta">${meta}</div>
      ${(document_.notes || []).map((note) =>
        `<div class="notice">${escapeHtml(note)}</div>`).join("")}
      ${textBlock("Extracted text", document_.raw_text)}
      ${document_.corrected_text && document_.corrected_text !== document_.raw_text
        ? textBlock("Corrected text", document_.corrected_text) : ""}
      ${document_.translated_text ? textBlock("Translation", document_.translated_text) : ""}
      ${!document_.raw_text ? '<div class="notice warn">No text was recognised in this file.</div>' : ""}
    </div>`;
}

/* ---------- library ---------- */

async function loadLibrary() {
  const list = $("library-list");
  const picker = $("ask-document");

  try {
    const documents = await request("/documents");

    picker.innerHTML =
      '<option value="">All documents</option>' +
      documents
        .map((item) => `<option value="${item.id}">${escapeHtml(item.filename)}</option>`)
        .join("");

    if (!documents.length) {
      list.innerHTML = '<p class="empty">Nothing stored yet. Process a document to begin.</p>';
      return;
    }

    list.innerHTML = `
      <div class="table-wrap"><table>
        <thead><tr>
          <th>ID</th><th>File</th><th>Languages</th>
          <th class="numeric">Chunks</th><th>Added</th><th></th>
        </tr></thead>
        <tbody>${documents.map(libraryRow).join("")}</tbody>
      </table></div>`;

    list.querySelectorAll("[data-view]").forEach((button) =>
      button.addEventListener("click", () => viewDocument(button.dataset.view))
    );
    list.querySelectorAll("[data-delete]").forEach((button) =>
      button.addEventListener("click", () => removeDocument(button.dataset.delete))
    );
  } catch (error) {
    list.innerHTML = errorNotice(error.message);
  }
}

function libraryRow(item) {
  const languages = item.detected_languages.map((language) => language.name).join(", ") || "—";
  return `
    <tr>
      <td>${item.id}</td>
      <td>${escapeHtml(item.filename)}</td>
      <td>${escapeHtml(languages)}</td>
      <td class="numeric">${item.chunk_count}</td>
      <td>${escapeHtml(item.created_at.replace("T", " ").replace("+00:00", ""))}</td>
      <td>
        <button class="button small" data-view="${item.id}">View</button>
        <button class="button small danger" data-delete="${item.id}">Delete</button>
      </td>
    </tr>`;
}

async function viewDocument(id) {
  const panel = $("document-detail");
  panel.hidden = false;
  panel.innerHTML = '<p class="empty">Loading…</p>';
  try {
    const document_ = await request(`/documents/${id}`);
    panel.innerHTML = renderDocument(document_, "Document");
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    panel.innerHTML = errorNotice(error.message);
  }
}

async function removeDocument(id) {
  if (!window.confirm(`Delete document ${id} and everything indexed from it?`)) return;
  try {
    await request(`/documents/${id}`, { method: "DELETE" });
    $("document-detail").hidden = true;
    loadLibrary();
    loadStatus();
  } catch (error) {
    $("library-list").innerHTML = errorNotice(error.message);
  }
}

$("refresh-library").addEventListener("click", loadLibrary);

/* ---------- ask ---------- */

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const output = $("ask-result");
  const chosen = $("ask-document").value;

  const payload = {
    question: $("question").value,
    top_k: Number($("ask-topk").value) || 3,
  };
  if (chosen) payload.document_ids = [Number(chosen)];

  const signal = askRunner.begin();
  output.innerHTML = "";

  try {
    const result = await request("/search/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    output.innerHTML = renderAnswer(result);
  } catch (error) {
    output.innerHTML = wasStopped(error) ? stoppedNotice : errorNotice(error.message);
  } finally {
    askRunner.end();
  }
});

function renderAnswer(result) {
  const sources = result.sources
    .map(
      (source, position) => `
      <div class="source">
        <div class="source-head">
          <span>Source ${position + 1} &middot; document ${source.document_id} &middot; chunk ${source.ordinal + 1}</span>
          <span>similarity ${source.score.toFixed(4)}</span>
        </div>
        <div class="source-text">${escapeHtml(source.text)}</div>
      </div>`
    )
    .join("");

  return `
    <div class="result">
      <h3>Answer</h3>
      <div class="answer${result.grounded ? "" : " ungrounded"}">${escapeHtml(result.answer)}</div>
      <div class="meta">
        <span class="pill ${result.grounded ? "ok" : "warn"}">${
          result.grounded ? "Grounded in sources" : "Not answered from sources"}</span>
        <span class="pill">Best similarity ${result.best_score.toFixed(4)}</span>
      </div>
      ${sources ? `<h3>Retrieved context</h3>${sources}` : ""}
    </div>`;
}

/* ---------- translate ---------- */

$("translate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const output = $("translate-result");

  const signal = translateRunner.begin();
  output.innerHTML = "";

  try {
    const result = await request("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: $("translate-input").value,
        target_language: $("translate-target").value || "English",
      }),
      signal,
    });

    output.innerHTML = `
      <div class="result">
        <h3>Translation</h3>
        <div class="meta">${languagePills(result.detected_languages)}</div>
        ${result.applied
          ? textBlock(result.target_language, result.translated_text)
          : errorNotice(`Translation failed: ${result.detail}`)}
      </div>`;
  } catch (error) {
    output.innerHTML = wasStopped(error) ? stoppedNotice : errorNotice(error.message);
  } finally {
    translateRunner.end();
  }
});

/* ---------- start ---------- */

const processRunner = createRunner("process");
const askRunner = createRunner("ask");
const translateRunner = createRunner("translate");

loadStatus();
loadLanguages();
loadLibrary();
setInterval(loadStatus, 20000);
