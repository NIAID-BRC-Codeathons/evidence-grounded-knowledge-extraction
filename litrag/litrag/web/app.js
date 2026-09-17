'use strict';

const $ = (id) => document.getElementById(id);
const state = { templates: [], lastBody: null, lastResult: null, glossary: null };

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
    const [templates, collections, backends, glossary] = await Promise.all([
      api('/api/templates'),
      api('/api/collections'),
      api('/api/backends'),
      api('/api/glossary'),
    ]);
    state.glossary = glossary;

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

function passageLinks(chunks) {
  // One link per retrieved passage. The chunk, not the paper, is what the
  // model actually read, so this is the evidence a curator needs to check.
  if (!chunks || !chunks.length) return '';
  return ' ' + chunks.map((ch) => {
    const span = (ch.start_char != null && ch.end_char != null)
      ? ` chars ${ch.start_char}\u2013${ch.end_char}` : '';
    const title = `Jump to retrieved passage [${ch.marker}]${span}`
      + (ch.chunk_id ? `\nchunk ${ch.chunk_id}` : '');
    return `<a class="passage" href="#source-${ch.marker}"`
      + ` data-marker="${ch.marker}" title="${escapeHtml(title)}">\u00b6${ch.marker}</a>`;
  }).join('');
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
    const paper = c.url
      ? `<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${text}</a>`
      : text;
    return `<span class="cite">${paper}${passageLinks(c.chunks)}</span>`;
  }).join('');
}

function linkMarkers(text, markerSet) {
  // Turn the "[3]" the model wrote inside an assertion into a jump to that
  // passage, so a claim can be checked against its source in one click.
  return escapeHtml(text).replace(/\[(\d+(?:\s*[,\u2013-]\s*\d+)?)\]/g, (whole, body) => {
    const parts = body.split(/[,\u2013-]/).map((n) => parseInt(n.trim(), 10))
      .filter((n) => !Number.isNaN(n));
    // "[1-3]" spans three passages; "[1, 2]" names two. Match how the
    // extractor reads them so the links agree with the resolved citations.
    const isSpan = parts.length === 2 && /[\u2013-]/.test(body);
    const expanded = isSpan
      ? Array.from({ length: Math.max(...parts) - Math.min(...parts) + 1 },
                   (_, i) => Math.min(...parts) + i)
      : parts;
    const nums = expanded.filter((n) => markerSet.has(n));
    if (!nums.length) return whole;
    return nums.map((n) => `<a class="marker" href="#source-${n}" data-marker="${n}"`
      + ` title="Jump to retrieved passage [${n}]">[${n}]</a>`).join('');
  });
}

function renderTable(result) {
  const dataColumns = result.columns.filter(
    (c) => !['reference', 'references', 'citation', 'citations', 'source'].includes(c.toLowerCase())
  );
  const template = currentTemplate() || {};
  const columnHelp = template.column_help || {};
  const derived = (state.glossary && state.glossary.derived) || {};

  const th = (label, help) => help
    ? `<th><span class="defined" data-help="${escapeHtml(help)}"`
      + ` tabindex="0" role="button" aria-label="${escapeHtml(label)}: ${escapeHtml(help)}">`
      + `${escapeHtml(label)}</span></th>`
    : `<th>${escapeHtml(label)}</th>`;

  const head = dataColumns.map((c) => th(c, columnHelp[c])).join('')
    + th('Support', derived.support)
    + th('Citations', derived.citations)
    + th('Flags', derived.flags);

  const body = result.rows.map((row) => {
    const markerSet = new Set(
      (row.citations || []).flatMap((c) => (c.chunks || []).map((ch) => ch.marker)));

    const cells = dataColumns.map((column) => {
      const raw = row.values[column] || '';
      // Any cell may carry a [n]; the Assertion usually does.
      let cell = linkMarkers(raw, markerSet);
      if (row.variants && row.variants[column]) {
        const others = row.variants[column].filter((v) => v !== row.values[column]);
        if (others.length) {
          cell += `<div class="variants">also reported: `
            + `${linkMarkers(others.join('; '), markerSet)}</div>`;
        }
      }
      return `<td>${cell}</td>`;
    }).join('');

    const support = `<td class="num"><span class="support">${row.n_support}</span></td>`;
    const cites = `<td>${citationHtml(row.citations)}</td>`;
    const flagHelp = (state.glossary && state.glossary.flags) || {};
    const flags = `<td>${row.flags.map((f) => {
      const help = flagHelp[f.split(':')[0]];
      return help
        ? `<span class="flag defined" data-help="${escapeHtml(help)}" tabindex="0"`
          + ` aria-label="${escapeHtml(f)}: ${escapeHtml(help)}">${escapeHtml(f)}</span>`
        : `<span class="flag">${escapeHtml(f)}</span>`;
    }).join('')}</td>`;
    return `<tr>${cells}${support}${cites}${flags}</tr>`;
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

    const span = (source.start_char != null && source.end_char != null)
      ? `<span class="metaTag">chars ${source.start_char}\u2013${source.end_char}</span>` : '';

    return `<div class="sourceCard" id="source-${index + 1}">
      <div class="sourceHead">
        <span class="rank">#${index + 1}</span>
        <span class="score">score ${Number(source.score).toFixed(3)}</span>
        ${corpus}
      </div>
      <div class="sourceTitle">${link}</div>
      <div>${meta}</div>
      ${authors ? `<div class="metaTag">${escapeHtml(authors)}</div>` : ''}
      <div class="chunkMeta">${span}<span class="metaTag chunkId" title="chunk id">`
        + `${escapeHtml(source.chunk_id || '')}</span></div>
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


function jumpToSource(marker) {
  const card = document.getElementById(`source-${marker}`);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  document.querySelectorAll('.sourceCard.highlight')
    .forEach((el) => el.classList.remove('highlight'));
  card.classList.add('highlight');
}


function showTip(target) {
  const text = target.dataset.help;
  if (!text) return;
  let tip = document.getElementById('tooltip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'tooltip';
    tip.className = 'tooltip';
    tip.setAttribute('role', 'tooltip');
    document.body.appendChild(tip);
  }
  tip.textContent = text;
  tip.style.visibility = 'hidden';
  tip.classList.add('visible');

  // Positioned against the viewport rather than nested in the table, whose
  // overflow-x container would otherwise clip it.
  const box = target.getBoundingClientRect();
  const tipBox = tip.getBoundingClientRect();
  const margin = 8;
  let left = box.left + box.width / 2 - tipBox.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - tipBox.width - margin));
  let top = box.bottom + 6;
  if (top + tipBox.height > window.innerHeight - margin) {
    top = box.top - tipBox.height - 6;
  }
  tip.style.left = `${left + window.scrollX}px`;
  tip.style.top = `${top + window.scrollY}px`;
  tip.style.visibility = 'visible';
}

function hideTip() {
  const tip = document.getElementById('tooltip');
  if (tip) tip.classList.remove('visible');
}


function init() {
  // Delegated so rebuilt tables keep working; focus included for keyboard use.
  document.addEventListener('mouseover', (e) => {
    const target = e.target.closest('.defined');
    if (target) showTip(target);
  });
  document.addEventListener('mouseout', (e) => {
    if (e.target.closest('.defined')) hideTip();
  });
  document.addEventListener('focusin', (e) => {
    const target = e.target.closest('.defined');
    if (target) showTip(target);
  });
  document.addEventListener('focusout', (e) => {
    if (e.target.closest('.defined')) hideTip();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideTip();
  });
  window.addEventListener('scroll', hideTip, { passive: true });
  // Delegated: the table and source list are rebuilt on every search.
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a.passage, a.marker');
    if (!link) return;
    e.preventDefault();
    jumpToSource(link.dataset.marker);
  });
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
