// Heuristic statistics view of the launcher page.
//
// Everything here is folded from what the run already wrote: the frozen
// catalogue and the corpus counts in viewer_index.json, plus the per-part
// summaries that ride on each part entry.  Individual verdict reasons are
// fetched from a part's heuristics.json only when a row is expanded, so the
// page stays cheap for a corpus that grows.
//
// No heuristic id appears in this file.  Rows, columns, filters and colours
// are derived from the data, so a heuristic added to the pipeline shows up
// here — including one whose domain is empty in the whole corpus, which is
// the coverage gap most worth seeing.

import { currentLine } from './line.js';
import { lineFolder, loadHeuristics } from './data.js';
import { externalGapsShown, onExternalGapsChange, setExternalGapsShown } from './prefs.js';

const OUTCOMES = ['PASS', 'FAIL', 'NOT_EVALUABLE', 'NOT_APPLICABLE'];
const CLASS = { PASS: 'pass', FAIL: 'fail', NOT_EVALUABLE: 'neval', NOT_APPLICABLE: 'na' };
// Same wording as the part page: the stored vocabulary is PASS / FAIL, the
// words on screen state whether the guideline holds, not whether the design
// is wrong.
const SHORT = { PASS: 'met', FAIL: 'not met', NOT_EVALUABLE: 'n/e', NOT_APPLICABLE: 'n/a' };
export function createStats(root, index) {
  // One dialect's numbers at a time.  A pass rate under v5.1 and one under
  // nx_v1 are not one metric (node 19a), so the two folds live side by side
  // in the index and the view reads the shown line's - never their sum, and
  // never one line's catalogue over another's verdicts.
  let catalogue = {};
  let corpus = {};
  let registry = {};
  let parts = [];
  let ids = [];

  const state = {
    scopes: new Set(),
    failsOnly: false,
    search: '',
    sortKey: 'hId',
    sortAsc: true,
    open: null,
    line: null,
    records: new Map(),   // partId -> heuristics.json, fetched on demand
  };

  /** The heuristic section of the line being shown, with the old shape as
   * the fallback so a run written before the split still renders. */
  function section() {
    const heuristics = index.heuristics || {};
    const lines = heuristics.lines || {};
    return lines[currentLine()] || (currentLine() === 'occ' ? heuristics : {});
  }

  /** One part's summary for the shown line; parts that line never built drop
   * out of the matrix rather than showing as empty rows. */
  function summaryOf(part) {
    const info = (part.lines && part.lines[currentLine()]) || null;
    if (info && info.heuristicSummary) return info.heuristicSummary;
    if (info && info.status && info.status !== 'PASS') return null;
    return currentLine() === 'occ' ? (part.heuristicSummary || null) : null;
  }

  function readLine() {
    const source = section();
    catalogue = source.catalogue || {};
    corpus = source.corpus || {};
    registry = source.registry || {};
    parts = (index.parts || []).filter((part) => summaryOf(part));
    ids = [...new Set([...Object.keys(catalogue), ...Object.keys(corpus)])].sort();
    state.records = new Map();
    state.open = null;
    state.line = currentLine();
  }

  readLine();

  /**
   * Counts as the reviewer asked to see them.
   *
   * Leaving out the external-input gaps is a display choice, not a recount of
   * the run: the verdicts still exist and the pipeline still owes nothing for
   * them, so they are subtracted here rather than anywhere upstream.
   */
  function counted(counts, blocked) {
    const gaps = (blocked || {}).externalInputMissing || 0;
    if (externalGapsShown() || !gaps) return { counts, hidden: 0 };
    return {
      counts: { ...counts, NOT_EVALUABLE: Math.max(0, (counts.NOT_EVALUABLE || 0) - gaps) },
      hidden: gaps,
    };
  }

  function entry(hId) {
    const source = corpus[hId] || {};
    const { counts, hidden } = counted(source.counts || {}, source.blocked);
    const definition = catalogue[hId] || {};
    const decided = (counts.PASS || 0) + (counts.FAIL || 0);
    return {
      hId,
      definition,
      counts,
      blocked: source.blocked || {},
      domainNodes: source.domainNodes || 0,
      records: Math.max(0, (source.records || 0) - hidden),
      decided,
      failRate: decided ? (counts.FAIL || 0) / decided : null,
    };
  }

  function visible() {
    const needle = state.search.trim().toLowerCase();
    return ids.map(entry).filter((row) => {
      const definition = row.definition;
      if (state.scopes.size && !state.scopes.has(definition.scope)) return false;
      if (state.failsOnly && !(row.counts.FAIL || 0)) return false;
      if (!needle) return true;
      return [row.hId, definition.labelEn, definition.statementEn]
        .filter(Boolean)
        .some((text) => String(text).toLowerCase().includes(needle));
    }).sort(compare);
  }

  function compare(a, b) {
    const direction = state.sortAsc ? 1 : -1;
    if (state.sortKey === 'hId') return a.hId.localeCompare(b.hId) * direction;
    if (state.sortKey === 'failRate') {
      const left = a.failRate === null ? -1 : a.failRate;
      const right = b.failRate === null ? -1 : b.failRate;
      return (left - right) * direction || a.hId.localeCompare(b.hId);
    }
    if (state.sortKey === 'domainNodes') {
      return (a.domainNodes - b.domainNodes) * direction || a.hId.localeCompare(b.hId);
    }
    return ((a.counts[state.sortKey] || 0) - (b.counts[state.sortKey] || 0)) * direction
      || a.hId.localeCompare(b.hId);
  }

  function render() {
    if (state.line !== currentLine()) readLine();
    root.innerHTML = '';
    root.append(cards(), matrixPanel(), rulePanel());
  }

  onExternalGapsChange(() => {
    if (!root.hidden) render();
  });

  // ----------------------------------------------------------------- cards

  function cards() {
    const rows = ids.map(entry);
    const totals = {};
    for (const outcome of OUTCOMES) {
      totals[outcome] = rows.reduce((sum, row) => sum + (row.counts[outcome] || 0), 0);
    }
    const emptyDomain = rows.filter((row) => !row.records).length;
    const decided = totals.PASS + totals.FAIL;

    // One figure.  Five equal cards gave the number a reader came for - the
    // met rate - exactly as much weight as the run's rule count, so nothing
    // led.  The run's other totals are counts you look up rather than scan, so
    // they sit in the card's tooltip alongside the rule and threshold hashes.
    const wrap = document.createElement('div');
    wrap.className = 'stats-summary';

    const hero = document.createElement('div');
    hero.className = 'stats-card stats-hero';
    const figure = document.createElement('div');
    figure.className = 'hero-figure';
    const big = document.createElement('div');
    big.className = 'value pass';
    big.textContent = decided ? `${Math.round((totals.PASS / decided) * 100)}%` : '-';
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = 'met, of the decided verdicts';
    figure.append(big, label);

    const split = document.createElement('div');
    split.className = 'hero-split';
    split.append(outcomeBar(totals, true));
    split.append(subLine(
      OUTCOMES.map((outcome) => `${SHORT[outcome]} ${totals[outcome]}`).join(' · '),
    ));
    hero.title = `${ids.length} heuristics evaluated`
      + ` · ${rows.reduce((sum, row) => sum + row.records, 0)} verdicts`
      + ` · ${rows.reduce((sum, row) => sum + row.domainNodes, 0)} instances in domain`
      + ` · ${emptyDomain} with no unit here`
      + `
rules ${registry.heuristicsYamlSha256 || '?'}`
      + `
thresholds ${registry.thresholdsYamlSha256 || '?'}`;
    hero.append(figure, split);
    wrap.append(hero);

    return wrap;
  }

  function subLine(text) {
    const line = document.createElement('div');
    line.className = 'sub';
    line.textContent = text;
    return line;
  }

  // ---------------------------------------------------------------- matrix

  function matrixPanel() {
    const panel = document.createElement('div');
    panel.className = 'stats-panel';
    const head = panelHead('Part × heuristic');
    head.append(legend());
    panel.append(head);

    const scroll = document.createElement('div');
    scroll.className = 'matrix-scroll';
    const table = document.createElement('table');
    table.className = 'matrix';

    const rows = visible();
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.append(document.createElement('th'));
    for (const row of rows) {
      const cell = document.createElement('th');
      cell.textContent = row.hId.replace('H', '');
      cell.title = `${row.hId} ${plainLabel(row.definition, row.hId)}`;
      headRow.append(cell);
    }
    thead.append(headRow);

    const tbody = document.createElement('tbody');
    for (const part of parts) {
      const line = document.createElement('tr');
      const name = document.createElement('th');
      name.textContent = part.partId;
      name.title = part.partId;
      line.append(name);
      for (const row of rows) {
        line.append(matrixCell(part, row, (summaryOf(part) || {})[row.hId]));
      }
      tbody.append(line);
    }

    table.append(thead, tbody);
    scroll.append(table);
    panel.append(scroll);
    return panel;
  }

  /** What the cell colours mean; the same classes the cells carry. */
  function legend() {
    const wrap = document.createElement('div');
    wrap.className = 'matrix-legend';
    for (const [cls, text, hint] of [
      ['fail', 'not met', 'at least one unit of the part does not meet the rule'],
      ['neval', 'not evaluable', 'no unit failed, at least one could not be decided'],
      ['pass', 'met', 'every evaluated unit meets the rule'],
      ['empty', 'no unit', 'the rule has no evaluated unit in this part'],
    ]) {
      const item = document.createElement('span');
      item.className = 'chip-solid ' + cls;
      item.textContent = text;
      item.title = hint;
      wrap.append(item);
    }
    return wrap;
  }

  function matrixCell(part, row, summary) {
    const cell = document.createElement('td');
    const button = document.createElement('a');
    button.className = 'cell';
    button.href = `viewer.html?part=${encodeURIComponent(part.partId)}&h=${encodeURIComponent(row.hId)}`;
    const { counts } = counted((summary && summary.counts) || {}, summary && summary.blocked);
    const decided = (counts.PASS || 0) + (counts.FAIL || 0) + (counts.NOT_EVALUABLE || 0);
    if (!summary || !decided) {
      button.classList.add('empty');
      button.title = `${part.partId} · ${row.hId}: no evaluated unit`;
    } else if (counts.FAIL) {
      button.classList.add('fail');
      button.title = `${part.partId} · ${row.hId}: ${counts.FAIL} not met of ${decided} evaluated`;
    } else if (counts.NOT_EVALUABLE) {
      button.classList.add('neval');
      button.title = `${part.partId} · ${row.hId}: ${counts.NOT_EVALUABLE} not evaluable`;
    } else {
      button.classList.add('pass');
      button.title = `${part.partId} · ${row.hId}: ${counts.PASS} met`;
    }
    cell.append(button);
    return cell;
  }

  // ------------------------------------------------------------ rule table

  function rulePanel() {
    const panel = document.createElement('div');
    panel.className = 'stats-panel';
    panel.append(panelHead('By heuristic'), filters(), table());
    return panel;
  }

  function panelHead(text) {
    const head = document.createElement('div');
    head.className = 'stats-panel-head';
    const heading = document.createElement('h2');
    heading.textContent = text;
    head.append(heading);
    return head;
  }

  function filters() {
    const bar = document.createElement('div');
    bar.className = 'stats-filters';

    const search = document.createElement('input');
    search.type = 'search';
    search.placeholder = 'search id, name or statement…';
    search.value = state.search;
    search.addEventListener('input', (event) => {
      state.search = event.target.value;
      const focus = document.activeElement === search;
      render();
      if (focus) {
        const next = root.querySelector('.stats-filters input[type="search"]');
        next.focus();
        next.setSelectionRange(next.value.length, next.value.length);
      }
    });
    bar.append(search);

    bar.append(chipGroup(
      'unit',
      [...new Set(ids.map((id) => (catalogue[id] || {}).scope).filter(Boolean))].sort(),
      state.scopes,
      (value) => ids.map(entry).filter((row) => row.definition.scope === value).length,
    ));

    bar.append(toggle('not met only', state.failsOnly, 'Keep only the heuristics with an unmet verdict', () => {
      state.failsOnly = !state.failsOnly;
      render();
    }));
    bar.append(toggle(
      'external',
      externalGapsShown(),
      'Also count the verdicts waiting on an input this pipeline does not own',
      () => setExternalGapsShown(!externalGapsShown()),
    ));
    return bar;
  }

  function chipGroup(label, values, selected, counter) {
    const group = document.createElement('span');
    group.className = 'group';
    const name = document.createElement('span');
    name.className = 'group-label';
    name.textContent = label;
    group.append(name);
    for (const value of values) {
      const chip = document.createElement('button');
      chip.className = 'type-chip' + (selected.has(value) ? ' on' : '');
      chip.textContent = `${value} ${counter(value)}`;
      chip.addEventListener('click', () => {
        if (selected.has(value)) selected.delete(value); else selected.add(value);
        render();
      });
      group.append(chip);
    }
    return group;
  }

  function toggle(label, on, hint, handler) {
    const button = document.createElement('button');
    button.className = 'switch compact';
    button.setAttribute('role', 'switch');
    button.setAttribute('aria-checked', String(Boolean(on)));
    button.title = hint;
    button.innerHTML = `<span class="switch-label">${label}</span>`
      + '<span class="switch-track"><span class="switch-thumb"></span></span>';
    button.addEventListener('click', handler);
    return button;
  }

  function table() {
    const wrap = document.createElement('div');
    const element = document.createElement('table');
    element.className = 'h-table';

    const columns = [
      { key: 'hId', label: 'H', className: 'col-id' },
      { key: null, label: 'name', className: 'col-name' },
      { key: null, label: 'outcomes', className: 'cell-bar', title: 'met / not met / not evaluable / not applicable — hover for the counts' },
      { key: 'domainNodes', label: 'domain', className: 'num', title: 'instances the rule talks about' },
      { key: 'failRate', label: 'not-met rate', className: 'num', title: 'not met ÷ (met + not met)' },
    ];

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const column of columns) {
      const cell = document.createElement('th');
      cell.textContent = column.label;
      cell.className = column.className;
      if (column.title) cell.title = column.title;
      if (column.key) {
        cell.dataset.sort = column.key;
        if (state.sortKey === column.key) {
          cell.classList.add('sorted');
          cell.dataset.direction = state.sortAsc ? 'asc' : 'desc';
        }
        cell.addEventListener('click', () => {
          if (state.sortKey === column.key) state.sortAsc = !state.sortAsc;
          else { state.sortKey = column.key; state.sortAsc = column.key === 'hId'; }
          render();
        });
      } else {
        cell.style.cursor = 'default';
      }
      headRow.append(cell);
    }
    thead.append(headRow);

    const tbody = document.createElement('tbody');
    for (const row of visible()) {
      tbody.append(ruleRow(row));
      if (state.open === row.hId) tbody.append(drillRow(row, columns.length));
    }

    element.append(thead, tbody);
    wrap.append(element);
    return wrap;
  }

  function ruleRow(row) {
    const element = document.createElement('tr');
    if (state.open === row.hId) element.classList.add('open');

    element.append(
      cell('col-id hid', row.hId),
      nameCell(row),
      barCell(row),
      numberCell(row.domainNodes),
      numberCell(row.failRate === null ? '—' : `${Math.round(row.failRate * 100)}%`,
        row.failRate ? 'fail' : ''),
    );
    element.addEventListener('click', () => {
      state.open = state.open === row.hId ? null : row.hId;
      render();
    });
    element.title = [
      row.definition.statementEn,
      row.definition.condition ? 'condition: ' + row.definition.condition : '',
      (row.definition.thresholds || [])
        .map((threshold) => `${threshold.symbol} = ${threshold.value}${threshold.unit ? ' ' + threshold.unit : ''}`)
        .join(', '),
      row.definition.externalDependency ? 'external input: ' + externalInput(row.definition) : '',
    ].filter(Boolean).join('\n');
    return element;
  }

  function nameCell(row) {
    const element = document.createElement('td');
    element.className = 'col-name hname';
    element.textContent = plainLabel(row.definition, row.hId);
    const sub = document.createElement('span');
    sub.className = 'en';
    sub.textContent = [row.definition.scope, externalInput(row.definition)].filter(Boolean).join(' · ');
    element.append(sub);
    return element;
  }

  function barCell(row) {
    const element = document.createElement('td');
    element.className = 'cell-bar';
    element.title = countsText(row.counts) || 'no evaluated unit';
    element.append(outcomeBar(row.counts));
    return element;
  }

  /** "met 78 · not met 17 · n/a 3" — the non-zero outcomes of a count table. */
  function countsText(counts) {
    return OUTCOMES
      .filter((outcome) => counts[outcome])
      .map((outcome) => `${SHORT[outcome]} ${counts[outcome]}`)
      .join(' · ');
  }

  function numberCell(value, tone) {
    const element = document.createElement('td');
    element.className = 'num ' + (value ? (tone || '') : 'zero');
    element.textContent = String(value);
    return element;
  }

  function cell(className, text) {
    const element = document.createElement('td');
    element.className = className;
    element.textContent = text;
    return element;
  }

  // ------------------------------------------------------------ drill-down

  function drillRow(row, span) {
    const element = document.createElement('tr');
    const container = document.createElement('td');
    container.colSpan = span;
    container.className = 'h-drill';

    const perPart = parts
      .map((part) => {
        const summary = (summaryOf(part) || {})[row.hId];
        const adjusted = counted((summary && summary.counts) || {}, summary && summary.blocked);
        return {
          part,
          counts: adjusted.counts,
          records: summary ? Math.max(0, summary.records - adjusted.hidden) : 0,
        };
      })
      .filter((item) => item.records);

    if (!perPart.length) {
      const empty = document.createElement('div');
      empty.className = 'insp-empty';
      empty.textContent = row.definition.appliesTo && row.definition.appliesTo.length
        ? `no ${row.definition.appliesTo.join(' / ')} instance in this run — empty domain`
        : 'no evaluated unit in this run';
      container.append(empty);
      element.append(container);
      return element;
    }

    const totals = document.createElement('div');
    totals.className = 'drill-line totals';
    totals.textContent = `${countsText(row.counts)} · ${row.domainNodes} in domain · ${perPart.length} part(s)`;
    container.append(totals);

    for (const { part, counts } of perPart) {
      const line = document.createElement('div');
      line.className = 'drill-line';

      const name = document.createElement('a');
      name.className = 'part';
      name.href = `viewer.html?part=${encodeURIComponent(part.partId)}&h=${encodeURIComponent(row.hId)}`;
      name.textContent = part.partId;
      line.append(name);
      line.append(outcomeBar(counts));

      const numbers = document.createElement('span');
      numbers.textContent = countsText(counts);
      line.append(numbers);
      container.append(line);

      if (counts.FAIL || counts.NOT_EVALUABLE) {
        const reasons = document.createElement('div');
        reasons.className = 'drill-reasons';
        container.append(reasons);
        fillReasons(reasons, part.partId, row.hId);
      }
    }

    element.append(container);
    return element;
  }

  async function fillReasons(container, partId, hId) {
    let payload = state.records.get(partId);
    if (!payload) {
      payload = await loadHeuristics(`data/${encodeURIComponent(partId)}/${lineFolder()}/`);
      state.records.set(partId, payload || { records: [] });
    }
    const rows = (payload.records || [])
      .filter((record) => record.hId === hId
        && record.outcome !== 'PASS'
        && record.outcome !== 'NOT_APPLICABLE'
        && !(!externalGapsShown() && (record.evidence || {}).blocked === 'externalInputMissing'))
      .slice(0, 8);
    container.innerHTML = '';
    for (const record of rows) {
      const line = document.createElement('a');
      line.className = 'heur-record ' + CLASS[record.outcome];
      line.href = `viewer.html?part=${encodeURIComponent(partId)}&h=${encodeURIComponent(hId)}`;
      const targets = [...(record.targets.nodeIds || []), ...(record.targets.edgeIds || [])];
      const name = document.createElement('span');
      name.className = 'heur-record-target';
      name.textContent = targets.length ? targets.join(' + ') : 'whole part';
      const reason = document.createElement('span');
      reason.className = 'heur-record-reason';
      reason.textContent = record.reason;
      line.append(name, reason);
      container.append(line);
    }
  }

  // ----------------------------------------------------------------- bits

  function outcomeBar(counts, tall) {
    const total = OUTCOMES.reduce((sum, outcome) => sum + (counts[outcome] || 0), 0);
    const bar = document.createElement('span');
    bar.className = 'outcome-bar' + (tall ? ' tall' : '');
    if (!total) return bar;
    for (const outcome of OUTCOMES) {
      const value = counts[outcome] || 0;
      if (!value) continue;
      const piece = document.createElement('span');
      piece.className = CLASS[outcome];
      piece.style.width = `${(value / total) * 100}%`;
      piece.title = `${SHORT[outcome]} ${value}`;
      bar.append(piece);
    }
    return bar;
  }

  /** The operand a heuristic waits for, named in the ontology's own terms. */
  function externalInput(definition) {
    if (!definition.externalDependency) return '';
    return definition.externalOperand || definition.externalDependency;
  }

  /** The frozen English label without its "H01 — " prefix. */
  function plainLabel(definition, hId) {
    const label = definition.labelEn || hId;
    return label.replace(/^H\d+\s*[—–-]\s*/, '');
  }

  return { render };
}
