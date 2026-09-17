'use strict';

const $ = (id) => document.getElementById(id);
const state = { templates: [], lastBody: null, lastResult: null };

function escapeHtml(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function currentTemplate() {
  return state.templates.find((t) => t.id === $('dataType').value) || null;
}

function requestBody() {
  return {
    organism: $('organism').value.trim(),
    genes: $('genes').value.trim(),
    other_terms: $('otherTerms').value.trim(),
    data_type: $('dataType').value,
    top_k: parseInt($('topK').value, 10),
    collection: $('collection').value || null,
    llm: $('backend').value,
    keep_empty: $('keepEmpty').checked,
    no_dedupe: !$('dedupe').checked,
  };
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload.detail) detail = payload.detail;
    } catch (err) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

async function loadMetadata() {
  try {
    const [templates, collections, backends] = await Promise.all([
      api('/api/templates'),
      api('/api/collections'),
      api('/api/backends'),
    ]);

    state.templates = templates.templates;
    const typeSelect = $('dataType');
    typeSelect.innerHTML = '';
    // Options come from the server, so a template added there appears here
    // with no change to this file.
    state.templates.forEach((t) => {
      const option = document.createElement('option');
      option.value = t.id;
      // No badge for locally-defined types: the default model is already a
      // local one, so the constraint only bites if the user picks the hosted
      // generator -- and checkBackendSupport explains it then.
      option.textContent = t.label + (t.output === 'table' ? '' : '  (prose)');
      typeSelect.appendChild(option);
    });
    const table = state.templates.find((t) => t.output === 'table');
    if (table) typeSelect.value = table.id;

    const collectionSelect = $('collection');
    collectionSelect.innerHTML = '';
    (collections.collections || []).forEach((c) => {
      const option = document.createElement('option');
      option.value = c.id;
      option.textContent = `${c.name} \u00b7 ${c.size_note}`;
      option.title = c.label || '';
      collectionSelect.appendChild(option);
    });
    if (collections.default) collectionSelect.value = collections.default;

    // The original widget's model dropdown was dead -- the hosted API exposes
    // no choice. Addressing the vLLM servers directly makes it real.
    const backendSelect = $('backend');
    backendSelect.innerHTML = '';
    (backends.backends || []).forEach((b) => {
      const option = document.createElement('option');
      option.value = b.id;
      option.textContent = b.label;
      option.title = b.note || '';
      backendSelect.appendChild(option);
    });
    if (backends.default) backendSelect.value = backends.default;
  } catch (err) {
    showError(`Could not reach the API: ${err.message}`);
  }

  try {
    const health = await api('/api/health');
    $('serverInfo').textContent =
      `litrag ${health.litrag} · ragstack ${health.server.version} (${health.server.git_tag})`;
  } catch (err) { /* header detail is optional */ }
}

function showError(message) {
  const status = $('status');
  status.className = 'status error';
  status.textContent = message;
  status.classList.remove('hidden');
}

function setBusy(busy) {
  $('searchBtn').disabled = busy;
  $('searchBtn').innerHTML = busy
    ? '<span class="spinner"></span>Searching…'
    : 'Search &amp; Extract';
}

function renderSummary(summary, rowCount) {
  const chips = [
    `<span class="chip"><strong>${rowCount}</strong> rows</span>`,
    `<span class="chip"><strong>${summary.n_sources}</strong> sources</span>`,
    `<span class="chip">${escapeHtml(summary.data_type)} v${summary.template_version}</span>`,
    `<span class="chip">${escapeHtml(summary.model || '')}</span>`,
    `<span class="chip">${escapeHtml(summary.generator || '')}</span>`,
    `<span class="chip">${escapeHtml(summary.collections || '')}</span>`,
    `<span class="chip">${summary.elapsed_s}s</span>`,
  ];
  if (summary.dropped_empty > 0) {
    chips.push(`<span class="chip warn">${summary.dropped_empty} evidence-free dropped</span>`);
  }
  if (summary.unresolved_citations > 0) {
    chips.push(`<span class="chip warn">${summary.unresolved_citations} unresolved citations</span>`);
  }
  const summaryEl = $('summary');
  summaryEl.innerHTML = chips.join('');
  summaryEl.classList.remove('hidden');
}

function citationHtml(citations) {
  if (!citations.length) return '<span class="flag">no citation</span>';
  return citations.map((c) => {
    if (!c.resolved) return `<span class="flag">[${c.marker}] unresolved</span>`;
    const label = [
      c.first_author ? `${c.first_author.split(' ').pop()} et al.` : null,
      c.journal, c.year ? `(${c.year})` : null,
    ].filter(Boolean).join(' ');
    const id = c.pmid ? `PMID ${c.pmid}` : (c.doi ? `DOI ${c.doi}` : '');
    const text = escapeHtml(`${label} ${id}`.trim()) || `[${c.marker}]`;
    return c.url
      ? `<span class="cite"><a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${text}</a></span>`
      : `<span class="cite">${text}</span>`;
  }).join('');
}

// Find `quote` inside `passage` and wrap it in <mark>, tolerating the
// whitespace and case drift a model introduces when copying. Returns escaped
// HTML. Falls back to the plain passage rather than guessing at a match.
function highlightQuote(passage, quote) {
  const safe = escapeHtml(passage);
  if (!quote) return safe;

  const escapedQuote = escapeHtml(quote).trim();
  const index = safe.indexOf(escapedQuote);
  if (index !== -1) {
    return safe.slice(0, index)
      + `<mark>${escapedQuote}</mark>`
      + safe.slice(index + escapedQuote.length);
  }

  // Whitespace-insensitive second pass: build a regex from the quote's words.
  const words = escapedQuote.split(/\s+/).filter(Boolean)
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (!words.length) return safe;
  try {
    const pattern = new RegExp(words.join('\\s+'), 'i');
    return safe.replace(pattern, (m) => `<mark>${m}</mark>`);
  } catch (err) {
    return safe;
  }
}

// The evidence panel: the passages this row was actually drawn from. This is
// the "evidence-grounded" claim made checkable -- a reader can see the source
// text rather than trusting the citation.
function evidenceHtml(row) {
  const withPassage = (row.citations || []).filter((c) => c.passage);
  const quote = (row.provenance && row.provenance.quote) || '';

  if (!withPassage.length) {
    return '<div class="evidenceEmpty">No passage recorded for this row. '
      + 'Runs made before passage provenance, or through the hosted path, '
      + 'carry citations without the supporting text.</div>';
  }

  const method = row.provenance && row.provenance.quote_method;
  const score = row.provenance && row.provenance.quote_score;
  const gate = method
    ? `<div class="gateNote">quote gate: <strong>${escapeHtml(method)}</strong>`
      + (score !== undefined && score !== null ? ` (score ${escapeHtml(String(score))})` : '')
      + '</div>'
    : '';

  return gate + withPassage.map((c) => {
    const ident = [
      c.pmid ? `PMID ${c.pmid}` : null,
      c.doi ? `DOI ${c.doi}` : null,
      c.chunk_id ? `chunk ${String(c.chunk_id).slice(0, 8)}` : null,
    ].filter(Boolean).join(' · ');
    return `<figure class="passage">
      <figcaption>
        <span class="passageTitle">${escapeHtml(c.title || 'Untitled')}</span>
        <span class="passageIds">${escapeHtml(ident)}</span>
      </figcaption>
      <blockquote>${highlightQuote(c.passage, quote)}</blockquote>
    </figure>`;
  }).join('');
}

function renderTable(result) {
  const dataColumns = result.columns.filter(
    (c) => !['reference', 'references', 'citation', 'citations', 'source'].includes(c.toLowerCase())
  );
  const head = '<th class="evCol"><span class="srOnly">Evidence</span></th>'
    + dataColumns.map((c) => `<th>${escapeHtml(c)}</th>`).join('')
    + '<th>Support</th><th>Citations</th><th>Flags</th>';

  const body = result.rows.map((row, rowIndex) => {
    const cells = dataColumns.map((column) => {
      let cell = escapeHtml(row.values[column] || '');
      if (row.variants && row.variants[column]) {
        const others = row.variants[column].filter((v) => v !== row.values[column]);
        if (others.length) {
          cell += `<div class="variants">also reported: ${escapeHtml(others.join('; '))}</div>`;
        }
      }
      return `<td>${cell}</td>`;
    }).join('');

    const support = `<td class="num"><span class="support">${row.n_support}</span></td>`;
    const cites = `<td>${citationHtml(row.citations)}</td>`;
    const flags = `<td>${row.flags.map((f) => `<span class="flag">${escapeHtml(f)}</span>`).join('')}</td>`;

    const hasEvidence = (row.citations || []).some((c) => c.passage);
    const toggle = `<td class="evCol">
      <button type="button" class="evToggle" data-row="${rowIndex}"
              aria-expanded="false" aria-controls="ev-${rowIndex}"
              title="${hasEvidence ? 'Show the supporting passage' : 'No passage recorded'}">
        <span aria-hidden="true">${hasEvidence ? '▸' : '·'}</span>
        <span class="srOnly">Show evidence for row ${rowIndex + 1}</span>
      </button></td>`;

    const span = dataColumns.length + 4;
    return `<tr class="dataRow">${toggle}${cells}${support}${cites}${flags}</tr>`
      + `<tr class="evidenceRow hidden" id="ev-${rowIndex}">`
      + `<td colspan="${span}">${evidenceHtml(row)}</td></tr>`;
  }).join('');

  return `<div class="tableWrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderSources(sources) {
  return sources.map((source, index) => {
    const meta = [
      source.journal ? escapeHtml(source.journal) : null,
      source.year ? escapeHtml(source.year) : null,
      source.pmid
        ? `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(source.pmid)}/" target="_blank" rel="noopener noreferrer">PMID ${escapeHtml(source.pmid)}</a>`
        : null,
      source.doi
        ? `<a href="https://doi.org/${escapeHtml(source.doi)}" target="_blank" rel="noopener noreferrer">DOI ${escapeHtml(source.doi)}</a>`
        : null,
    ].filter(Boolean).map((m) => `<span class="metaTag">${m}</span>`).join('');

    const authors = (source.authors || []).slice(0, 3).join(', ')
      + ((source.authors || []).length > 3 ? ', et al.' : '');
    const content = (source.content || '').replace(/\s+/g, ' ').trim();
    const isLong = content.length > 400;
    const preview = isLong ? content.slice(0, 400) + '…' : content;
    const title = escapeHtml(source.title || 'Untitled');
    const link = source.doi
      ? `<a href="https://doi.org/${escapeHtml(source.doi)}" target="_blank" rel="noopener noreferrer">${title}</a>`
      : title;

    const corpus = source.collection
      ? `<span class="corpusTag">${escapeHtml(source.collection)}</span>` : '';

    return `<div class="sourceCard">
      <div class="sourceHead">
        <span class="rank">#${index + 1}</span>
        <span class="score">score ${Number(source.score).toFixed(3)}</span>
        ${corpus}
      </div>
      <div class="sourceTitle">${link}</div>
      <div>${meta}</div>
      ${authors ? `<div class="metaTag">${escapeHtml(authors)}</div>` : ''}
      <div class="sourceContent" data-full="${escapeHtml(content)}" data-preview="${escapeHtml(preview)}">${escapeHtml(preview)}</div>
      ${isLong ? '<button type="button" class="expandBtn">Show more</button>' : ''}
    </div>`;
  }).join('');
}

function renderResult(result) {
  state.lastResult = result;
  renderSummary(result.summary, result.rows.length);

  $('answerTitle').textContent = result.is_table ? 'Extracted Data' : 'Summary';
  $('answerBody').innerHTML = result.is_table
    ? (result.rows.length
        ? renderTable(result)
        : '<div class="status">No rows extracted. Try a broader query or more RAG results.</div>')
    : `<div class="answerText">${escapeHtml(result.answer)}</div>`;
  $('answerSection').classList.remove('hidden');

  $('sourcesTitle').textContent = `Sources (${result.sources.length})`;
  $('sourcesBody').innerHTML = renderSources(result.sources);
  $('sourcesSection').classList.remove('hidden');
  $('status').classList.add('hidden');
  $('viewRequestBtn').disabled = false;
}

async function search(event) {
  event.preventDefault();
  const body = requestBody();
  if (!body.organism) { $('organism').focus(); return; }

  state.lastBody = body;
  setBusy(true);
  $('status').className = 'status';
  $('status').innerHTML = '<span class="spinner"></span>Searching literature and extracting…';
  $('status').classList.remove('hidden');
  $('answerSection').classList.add('hidden');
  $('sourcesSection').classList.add('hidden');
  $('summary').classList.add('hidden');

  try {
    const result = await api('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    renderResult(result);
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

async function download(fmt) {
  if (!state.lastBody) return;
  try {
    const response = await fetch(`/api/export?fmt=${encodeURIComponent(fmt)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.lastBody),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `litrag.${fmt === 'table' ? 'txt' : fmt}`;
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showError(`Download failed: ${err.message}`);
  }
}

async function viewRequest() {
  const body = state.lastBody || requestBody();
  try {
    const preview = await api('/api/request', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    let text = `POST ${preview.endpoint}\n\n${JSON.stringify(preview.body, null, 2)}\n\n`
      + `--- template declaration ---\n${JSON.stringify(preview.template, null, 2)}`;

    if (preview.prompt) {
      // Generating locally means LitRAG owns the prompt, so unlike the hosted
      // path it can actually be shown.
      $('modalNote').textContent =
        'Generating locally, so this is the full prompt LitRAG sends '
        + `(hash ${preview.prompt_hash}). Retrieved context is omitted here.`;
      text += `\n\n--- prompt ---\n${preview.prompt}`;
    } else {
      $('modalNote').textContent =
        'The hosted path keeps its prompt server-side and the API does not expose it. '
        + 'This is the exact request sent, with the template version and hash.';
    }
    $('modalBody').textContent = text;
    $('modal').classList.remove('hidden');
  } catch (err) {
    showError(err.message);
  }
}

function checkBackendSupport() {
  const template = currentTemplate();
  const isServer = $('backend').value === 'server';
  if (template && template.local && isServer) {
    showError(`"${template.label}" is defined by LitRAG, not the server, `
      + 'so it needs a local model. Pick Qwen or Llama.');
    $('searchBtn').disabled = true;
  } else {
    $('searchBtn').disabled = false;
    if ($('status').classList.contains('error')) {
      $('status').className = 'status';
      $('status').textContent = 'Enter an organism to search the literature.';
    }
  }
}


function init() {
  $('dataType').addEventListener('change', checkBackendSupport);
  $('backend').addEventListener('change', checkBackendSupport);
  $('topK').addEventListener('input', (e) => { $('topKValue').textContent = e.target.value; });
  $('searchForm').addEventListener('submit', search);
  $('viewRequestBtn').addEventListener('click', viewRequest);
  $('modalClose').addEventListener('click', () => $('modal').classList.add('hidden'));
  $('modal').addEventListener('click', (e) => {
    if (e.target === $('modal')) $('modal').classList.add('hidden');
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $('modal').classList.add('hidden');
  });
  document.querySelectorAll('button.dl').forEach((button) => {
    button.addEventListener('click', () => download(button.dataset.fmt));
  });
  // Source cards are rebuilt on every search, so delegate instead of rebinding.
  // Evidence drill-down. Delegated, because the table is re-rendered wholesale
  // on every search and per-row listeners would be rebound each time.
  $('answerBody').addEventListener('click', (e) => {
    const button = e.target.closest('.evToggle');
    if (!button) return;
    const panel = document.getElementById(`ev-${button.dataset.row}`);
    if (!panel) return;
    const open = !panel.classList.toggle('hidden');
    button.setAttribute('aria-expanded', String(open));
    const caret = button.querySelector('[aria-hidden]');
    if (caret && caret.textContent !== '·') caret.textContent = open ? '▾' : '▸';
  });

  $('sourcesBody').addEventListener('click', (e) => {
    if (!e.target.classList.contains('expandBtn')) return;
    const content = e.target.previousElementSibling;
    const expanded = e.target.textContent === 'Show less';
    content.textContent = expanded ? content.dataset.preview : content.dataset.full;
    e.target.textContent = expanded ? 'Show more' : 'Show less';
  });
  loadMetadata();
}

document.addEventListener('DOMContentLoaded', init);
