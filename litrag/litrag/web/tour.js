'use strict';

/* A guided tour that walks the app, not just the page.
 *
 * Mechanism follows the Evidence Desk prototype:
 *
 *   - A spotlight cut out of a dimming layer, made with one enormous spread
 *     shadow on a transparent bordered box. The earlier version lifted the
 *     target above an overlay with z-index and forced a white background,
 *     which broke on anything non-opaque and inverted badly in dark mode.
 *   - Progress dots beside a step count, and an always-available Skip.
 *   - Each step may carry an `enter` that puts the app into the state the step
 *     is about. Without that a tour can only describe controls that happen to
 *     be on screen -- and the two things most worth showing here, the evidence
 *     panel and the prompt lab, do not exist on a freshly loaded page.
 *
 * The demo row is a recorded fixture, not a live query. A tour that waits 30
 * seconds for a gateway round trip is a tour nobody finishes, and one that
 * fails when the network is down is worse than none at a venue.
 */

(function () {
  const SEEN_KEY = 'litrag.tour.seen';
  const DIM = 'rgba(22, 32, 42, 0.55)';

  // One recorded row, enough to render the table and an evidence panel. Taken
  // from a real katG run: gpt56sol, quote gate on, matched exact.
  const DEMO = {
    summary: {
      n_sources: 5, data_type: 'mutation', template_version: 2,
      model: 'gpt-5.6-sol', backend: 'argo (demo data)', generator: 'local',
      collections: 'open-access', elapsed_s: 0, dropped_empty: 0,
      unresolved_citations: 0, prompt_hash: 'demo000000000000',
    },
    is_table: true,
    columns: ['Organism', 'Gene Name', 'Mutation', 'Phenotype', 'Assertion', 'Reference'],
    answer: '',
    request_body: {},
    sources: [],
    rows: [{
      values: {
        Organism: 'Mycobacterium tuberculosis', 'Gene Name': 'katG',
        Mutation: 'S315T', Phenotype: 'Moderate-level isoniazid resistance',
        Assertion: 'The most common mutation conferring isoniazid resistance.',
        Reference: '[1]',
      },
      flags: [], variants: {}, n_support: 4,
      provenance: { quote: 'the most common mutation that confers resistance to the prodrug isoniazid', quote_method: 'exact', quote_score: 1.0 },
      citations: [{
        marker: 1, pmid: '40580943', pmcid: null, doi: '10.1093/gbe/evaf120',
        journal: 'Genome Biology and Evolution', year: '2025',
        title: 'Fitness Effect of the Isoniazid Resistance Mutation S315T of KatG',
        first_author: 'Ugo Bastolla', resolved: true, matched_by: 'marker',
        chunk_id: '351cd7ac-49f3-564c-ad70-e5b8de8e464b',
        doc_id: '33147298-cccf-53f7-a75b-283dc4220b31',
        passage: 'Abstract The mutation S315T of the catalase-peroxidase (CP) protein '
               + 'KatG of Mycobacterium tuberculosis is the most common mutation that '
               + 'confers resistance to the prodrug isoniazid. Here, we reconstruct its '
               + 'evolutionary history in 145 whole-genome sequences of M. tuberculosis, '
               + 'inferring 11 independent appearances of this mutation and 5 reversion '
               + 'events.',
      }],
    }],
  };

  const STEPS = [
    {
      target: '#organism',
      title: 'Start with an organism',
      body: 'Every query is anchored to one organism. Genes and other terms narrow '
          + 'it; leave them blank to cast wider.',
    },
    {
      target: '#dataType',
      title: 'Pick what kind of fact you want',
      body: 'Each data type is a different table shape. Alongside mutations and '
          + 'protein interactions are the three relation types this project '
          + 'targets: pathogen-host phenotype, pathogen-mechanism-disease and '
          + 'biomarker-disease.',
    },
    {
      target: '#backend',
      title: 'Choose who reads the passages',
      body: 'Retrieval always comes from the literature index. This picks the model '
          + 'that reads what was retrieved and fills the table. The Argonne gateway '
          + 'serves many models and will not guess, so a model box appears beside it.',
    },
    {
      target: '#quoteGate',
      title: 'Cite or refuse',
      body: 'With this on, the model must copy a verbatim sentence supporting each '
          + 'row, and any row whose quote is not actually in the cited passage is '
          + 'dropped and counted rather than quietly kept.',
    },
    {
      target: '.evToggle',
      title: 'Every row shows its evidence',
      body: 'This is a recorded example, not a live run. Expand any row to see the '
            + 'passage it came from, the supporting sentence highlighted, and the '
            + 'gate verdict. A claim you can check is the whole point.',
      enter: showDemo,
      spotPad: 4,
    },
    {
      target: '#promptLab',
      title: 'Prompt lab',
      body: 'Run one query under two system prompts and diff the rows. Good for '
          + 'forming hypotheses — but these models are nondeterministic, so a '
          + 'difference between two single runs is not evidence that either prompt '
          + 'is better.',
      enter: () => { const lab = document.getElementById('promptLab'); if (lab) lab.open = true; },
    },
  ];

  let index = 0;
  let overlay = null;
  let spot = null;
  let card = null;
  let demoShown = false;

  function showDemo() {
    if (demoShown) return;
    // renderResult lives in app.js and is the same path a real query takes, so
    // the tour shows the actual component rather than a mock of it.
    if (typeof renderResult === 'function') {
      renderResult(JSON.parse(JSON.stringify(DEMO)));
      const first = document.querySelector('.evToggle');
      if (first && first.getAttribute('aria-expanded') !== 'true') first.click();
      demoShown = true;
    }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function placeSpot(target, pad) {
    const box = target.getBoundingClientRect();
    const p = pad === undefined ? 6 : pad;
    Object.assign(spot.style, {
      left: `${box.left + window.scrollX - p}px`,
      top: `${box.top + window.scrollY - p}px`,
      width: `${box.width + p * 2}px`,
      height: `${box.height + p * 2}px`,
      // The spotlight: the box is transparent, and one very large spread
      // shadow dims everything outside it.
      boxShadow: `0 0 0 9999px ${DIM}`,
    });
  }

  function placeCard(target) {
    const box = target.getBoundingClientRect();
    const margin = 14;
    card.style.visibility = 'hidden';
    card.style.top = '0px';
    const height = card.offsetHeight;
    const width = card.offsetWidth;

    const below = box.bottom + margin + height < window.innerHeight;
    const top = below ? box.bottom + margin
                      : Math.max(margin, box.top - height - margin);
    let left = box.left;
    if (left + width > window.innerWidth - margin) {
      left = Math.max(margin, window.innerWidth - width - margin);
    }
    card.style.top = `${top + window.scrollY}px`;
    card.style.left = `${left + window.scrollX}px`;
    card.style.visibility = 'visible';
  }

  function show(stepIndex) {
    const step = STEPS[stepIndex];
    if (step.enter) {
      try { step.enter(); } catch (err) { /* a step must never wedge the tour */ }
    }

    const target = document.querySelector(step.target);
    if (!target) { advance(); return; }

    target.scrollIntoView({ block: 'center', behavior: 'smooth' });

    card.innerHTML = '';

    const head = el('div', 'tourHead');
    head.appendChild(el('span', 'tourCount', `${stepIndex + 1} / ${STEPS.length}`));
    const skip = el('button', 'tourSkip', 'Skip');
    skip.type = 'button';
    skip.addEventListener('click', stop);
    head.appendChild(skip);
    card.appendChild(head);

    card.appendChild(el('h3', 'tourTitle', step.title));
    card.appendChild(el('p', 'tourBody', step.body));

    const footer = el('div', 'tourFooter');
    if (stepIndex > 0) {
      const back = el('button', 'tourBtn', 'Back');
      back.type = 'button';
      back.addEventListener('click', () => show(--index));
      footer.appendChild(back);
    }
    const forward = el('button', 'tourBtn primary',
      stepIndex === STEPS.length - 1 ? 'Done' : 'Next');
    forward.type = 'button';
    forward.addEventListener('click', advance);
    footer.appendChild(forward);

    const dots = el('div', 'tourDots');
    STEPS.forEach((_, i) => {
      const dot = el('span', 'tourDot' + (i === stepIndex ? ' on' : ''));
      dots.appendChild(dot);
    });
    footer.appendChild(dots);
    card.appendChild(footer);

    // Layout settles after the scroll, so place on the next frame.
    requestAnimationFrame(() => { placeSpot(target, step.spotPad); placeCard(target); });
    forward.focus();
  }

  function advance() {
    index += 1;
    if (index >= STEPS.length) { stop(); return; }
    show(index);
  }

  function stop() {
    [overlay, spot, card].forEach((node) => node && node.remove());
    overlay = spot = card = null;
    try { localStorage.setItem(SEEN_KEY, '1'); } catch (err) { /* private mode */ }
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', reposition);
  }

  function reposition() {
    if (!card) return;
    const target = document.querySelector(STEPS[index].target);
    if (target) { placeSpot(target, STEPS[index].spotPad); placeCard(target); }
  }

  function onKey(event) {
    if (event.key === 'Escape') stop();
    else if (event.key === 'ArrowRight') advance();
    else if (event.key === 'ArrowLeft' && index > 0) show(--index);
  }

  function start() {
    if (card) return;
    index = 0;
    demoShown = false;

    overlay = el('div', 'tourOverlay');
    overlay.addEventListener('click', stop);
    spot = el('div', 'tourSpot');
    card = el('div', 'tourCard');
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-modal', 'true');
    card.setAttribute('aria-label', 'Guided tour');

    document.body.appendChild(overlay);
    document.body.appendChild(spot);
    document.body.appendChild(card);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', reposition);
    show(0);
  }

  window.litragTour = { start, stop };

  document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('tourBtn');
    if (button) button.addEventListener('click', start);

    let seen = true;
    try { seen = localStorage.getItem(SEEN_KEY) === '1'; } catch (err) { seen = true; }
    if (!seen) setTimeout(start, 1200);
  });
}());
