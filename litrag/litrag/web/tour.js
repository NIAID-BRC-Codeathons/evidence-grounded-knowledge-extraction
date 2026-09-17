'use strict';

/* A first-run guided tour.
 *
 * LitRAG's form looks like a search box, which undersells it and mis-sets
 * expectations: the interesting parts are the relation types, the evidence
 * panel and the prompt lab, none of which announce themselves. A short tour is
 * cheaper than documentation nobody opens.
 *
 * Deliberately dependency-free and ~150 lines. A tour library would be larger
 * than the UI it explains.
 */

(function () {
  const SEEN_KEY = 'litrag.tour.seen';

  const STEPS = [
    {
      target: '#organism',
      title: 'Start with an organism',
      body: 'Every query is anchored to one organism. Genes and other terms '
          + 'narrow it — leave them blank to cast wider.',
    },
    {
      target: '#dataType',
      title: 'Pick what kind of fact you want',
      body: 'Each data type is a different table shape. Alongside mutations and '
          + 'protein interactions you will find the three relation types this '
          + 'project targets: pathogen-host phenotype, pathogen-mechanism-disease '
          + 'and biomarker-disease.',
    },
    {
      target: '#backend',
      title: 'Choose who does the extraction',
      body: 'Retrieval always comes from the literature index. This picks the '
          + 'model that reads the passages and fills the table. The Argonne '
          + 'gateway needs a named model, so a model box appears beside it.',
    },
    {
      target: '#quoteGate',
      title: 'Cite or refuse',
      body: 'With this on, the model must copy a verbatim sentence supporting '
          + 'each row. Any row whose quote is not actually in the cited passage '
          + 'is dropped and counted, rather than quietly kept.',
    },
    {
      target: '#searchBtn',
      title: 'Run it',
      body: 'Expect 20-60 seconds. Every result row gets a ▸ expander showing '
          + 'the exact passage it came from, with the supporting sentence '
          + 'highlighted — that is the point of the tool: you can check it.',
    },
    {
      target: '#promptLab',
      title: 'Prompt lab',
      body: 'Run the same query under two system prompts and diff the rows. '
          + 'Useful for forming hypotheses — but these models are '
          + 'nondeterministic, so a difference between two single runs is not '
          + 'evidence that one prompt is better.',
      open: true,
    },
  ];

  let index = 0;
  let overlay = null;
  let card = null;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function position(target) {
    const box = target.getBoundingClientRect();
    const margin = 12;
    card.style.visibility = 'hidden';
    card.style.top = '0px';
    const height = card.offsetHeight;
    const width = card.offsetWidth;

    // Below the target when there is room, otherwise above it.
    const below = box.bottom + margin + height < window.innerHeight;
    const top = below ? box.bottom + margin : Math.max(margin, box.top - height - margin);
    let left = box.left;
    if (left + width > window.innerWidth - margin) {
      left = Math.max(margin, window.innerWidth - width - margin);
    }

    card.style.top = `${top + window.scrollY}px`;
    card.style.left = `${left + window.scrollX}px`;
    card.style.visibility = 'visible';
  }

  function clearHighlight() {
    document.querySelectorAll('.tourTarget')
      .forEach((n) => n.classList.remove('tourTarget'));
  }

  function show(stepIndex) {
    const step = STEPS[stepIndex];
    const target = document.querySelector(step.target);
    if (!target) { next(); return; }

    if (step.open && target.tagName === 'DETAILS') target.open = true;

    clearHighlight();
    target.classList.add('tourTarget');
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });

    card.innerHTML = '';
    card.appendChild(el('h3', 'tourTitle', step.title));
    card.appendChild(el('p', 'tourBody', step.body));

    const footer = el('div', 'tourFooter');
    footer.appendChild(el('span', 'tourCount', `${stepIndex + 1} of ${STEPS.length}`));

    const spacer = el('span', 'tourSpacer');
    footer.appendChild(spacer);

    if (stepIndex > 0) {
      const back = el('button', 'tourBtn', 'Back');
      back.type = 'button';
      back.addEventListener('click', () => show(--index));
      footer.appendChild(back);
    }
    const forward = el('button', 'tourBtn primary',
      stepIndex === STEPS.length - 1 ? 'Done' : 'Next');
    forward.type = 'button';
    forward.addEventListener('click', next);
    footer.appendChild(forward);

    card.appendChild(footer);
    position(target);
    forward.focus();
  }

  function next() {
    index += 1;
    if (index >= STEPS.length) { stop(); return; }
    show(index);
  }

  function stop() {
    clearHighlight();
    if (overlay) overlay.remove();
    if (card) card.remove();
    overlay = null;
    card = null;
    try { localStorage.setItem(SEEN_KEY, '1'); } catch (err) { /* private mode */ }
    document.removeEventListener('keydown', onKey);
  }

  function onKey(event) {
    if (event.key === 'Escape') stop();
    if (event.key === 'ArrowRight') next();
  }

  function start() {
    if (card) return;
    index = 0;
    overlay = el('div', 'tourOverlay');
    overlay.addEventListener('click', stop);
    card = el('div', 'tourCard');
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'Guided tour');
    document.body.appendChild(overlay);
    document.body.appendChild(card);
    document.addEventListener('keydown', onKey);
    show(0);
  }

  window.litragTour = { start, stop };

  document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('tourBtn');
    if (button) button.addEventListener('click', start);

    let seen = true;
    try { seen = localStorage.getItem(SEEN_KEY) === '1'; } catch (err) { seen = true; }
    // Wait for the dropdowns to populate, so step 2 points at a filled control.
    if (!seen) setTimeout(start, 1200);
  });
}());
