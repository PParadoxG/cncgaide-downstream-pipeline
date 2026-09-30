// Store state on the part list: status chips, row actions, the job bar and
// the add-parts flow.
//
// Everything here is driven by data the server already has — `api/parts`
// rows (the light, both lines) and `api/status` (current job, queue,
// history).  No feature, relation or heuristic name is known to this file; a
// changed layer is shown by the name the store gives it.

import { escapeHtml } from './html.js';
import { deletePart, describeJob, postRefresh, safeStepName, uploadStep } from './refresh.js';

// The light -> what the chip says.  The light is about the source STEP and
// nothing else; `queued` / `building` come from the job queue and win over it
// while they apply, and `rules` is not a light at all but the run-wide "this
// part was built under older rules" state, which colours every affected row.
const CHIP = {
  fresh: { label: 'fresh', cls: 'fresh' },
  changed: { label: 'source changed', cls: 'stale' },
  source_missing: { label: 'source missing', cls: 'missing' },
  untracked: { label: 'untracked', cls: 'missing' },
  new: { label: 'not built', cls: 'new' },
  rules: { label: 'rules changed', cls: 'rules' },
  queued: { label: 'queued', cls: 'queued' },
  building: { label: 'building…', cls: 'building' },
  removing: { label: 'removing…', cls: 'building' },
};

// What one line's own state says, when the light has nothing to report.  PASS
// is the rule and stays quiet; the other two are exceptions worth a chip.
// NOT_BUILT is deliberately not `failed`: it is about this machine, not about
// this part, and the next refresh retries it.
const LINE_CHIP = {
  FAIL: { label: 'failed', cls: 'failed' },
  NOT_BUILT: { label: 'not computed', cls: 'missing' },
};

/** The chip state of one `api/parts` row (null row = no store information). */
export function chipState(row) {
  if (!row) return null;
  if (row.job) {
    if (row.job.kind === 'remove') return 'removing';
    return row.job.state === 'running' ? 'building' : 'queued';
  }
  const light = (row.light && row.light.state) || 'fresh';
  if (light === 'fresh' && (row.rulesChanged || []).length) return 'rules';
  return light;
}

/**
 * The status cell of one row, read through *line* (the line the page is
 * currently showing).  Three questions in priority order, because they answer
 * different things and only the first one that applies is worth a chip:
 * is a job running, has the source moved (or the rules), and finally — what
 * did this line itself manage to do.
 */
export function statusChip(row, line) {
  const wrap = document.createElement('span');
  wrap.className = 'col-status';
  const state = chipState(row);
  if (!state) {
    wrap.innerHTML = '<span class="st-chip unknown" title="store status unavailable">—</span>';
    return wrap;
  }
  const tips = [];
  let spec = null;
  let label = '';

  if (state === 'fresh') {
    // The source is unchanged and the rules are current: whatever is left to
    // say belongs to this line.  A part that is simply up to date is the
    // rule, not an exception, and spends no colour saying so.
    const info = (row.lines || {})[line] || null;
    const status = info && info.status;
    if (status === 'PASS' || (!info && !row.error)) {
      wrap.title = info && info.producedAt
        ? `up to date · built ${info.producedAt} (UTC)`
        : 'up to date with the store';
      return wrap;
    }
    spec = LINE_CHIP[status] || { label: 'not built', cls: 'new' };
    label = spec.label;
    if (info && info.reason) tips.push(info.reason);
    if (info && info.error) tips.push(info.error);
  } else {
    spec = CHIP[state] || { label: state, cls: 'stale' };
    label = spec.label;
    if (state === 'rules') {
      const layers = row.rulesChanged || [];
      label += ' · ' + layers.join(', ');
      tips.push(`built under older ${layers.join('/')} code — refresh to rebuild`);
    }
    for (const reason of (row.light && row.light.reasons) || []) tips.push(reason);
  }

  if (row.error) tips.push(row.error);
  if (row.job) tips.push(`${row.job.jobId} ${row.job.state}`);
  const chip = document.createElement('span');
  chip.className = `st-chip ${spec.cls}`;
  chip.dataset.state = state;
  chip.textContent = label;
  chip.title = tips.join('\n');
  wrap.append(chip);
  return wrap;
}

export function metaCell(row, producedAt) {
  const span = document.createElement('span');
  span.className = 'col-meta';
  if (!row && producedAt === undefined) { span.textContent = ''; return span; }
  // Build time only.  The source fingerprint is what the store compares to
  // decide staleness, not something a reader scans a column of; it stays in
  // the cell's tooltip, where it is there when a hash is actually wanted.
  // *producedAt* is the shown line's build time; the row itself has none,
  // because "when was this part built" is a question per line.
  const shown = producedAt !== undefined ? producedAt : (row && row.producedAt);
  const when = shown ? shortTime(shown) : '—';
  span.innerHTML = `<span class="meta-when">${escapeHtml(when)}</span>`;
  span.title = `produced at ${shown || '—'} (UTC)`
    + (row && row.fingerprint ? `\nsource STEP sha256 ${row.fingerprint}` : '');
  return span;
}

function shortTime(iso) {
  // 2026-08-22T02:10:31Z -> 08-22 02:10
  const match = /^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/.exec(iso);
  return match ? `${match[1]} ${match[2]}` : iso;
}

/**
 * Row actions.  `handlers.rerun(partId)` / `handlers.remove(partId)`; the
 * buttons are disabled while the part has a job or the server is read-only.
 */
export function actionsCell(row, partId, handlers, enabled) {
  const wrap = document.createElement('span');
  wrap.className = 'col-actions';
  const busy = Boolean(row && row.job);
  const rerun = button('Re-run', 'rebuild this part now (refresh --only)', () => handlers.rerun(partId));
  const remove = button('Remove', 'drop this part from the store', () => handlers.remove(partId));
  rerun.disabled = !enabled || busy;
  const unbuilt = row?.light?.state === 'new';
  remove.disabled = !enabled || busy || !row || unbuilt;
  if (unbuilt) rerun.textContent = 'Build';
  // Similarity needs a built part; it is a read action, so it stays
  // available on a read-only server.
  if (handlers.similar && row && row.status === 'PASS') {
    wrap.append(button('Similar', 'rank the other parts by graph edit distance to this one', () => handlers.similar(partId)));
  }
  wrap.append(rerun, remove);
  return wrap;
}

function button(text, title, onClick) {
  const element = document.createElement('button');
  element.type = 'button';
  element.className = 'btn-text';
  element.textContent = text;
  element.title = title;
  element.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    onClick();
  });
  return element;
}

// ----------------------------------------------------------------- actions

// "Re-run" means rebuild, also for a part the store considers fresh; that is
// mode "all" narrowed to one id.  (mode "stale" would be a no-op for it.)
export async function rerunPart(partId) {
  return postRefresh({ mode: 'all', only: [partId] });
}

/**
 * Two plain confirms rather than a dialog: the second question matters,
 * because a part whose STEP stays in the input directory comes straight back
 * on the next refresh.
 */
export async function removePartInteractive(partId) {
  const proceed = window.confirm(
    `Remove ${partId} from the store?\n\nIts results under runs/ are deleted. `
    + 'The STEP file stays in the input directory unless you also delete it in the next step.',
  );
  if (!proceed) return null;
  const source = window.confirm(
    `Also delete the source STEP of ${partId} from the input directory?\n\n`
    + 'OK = delete the file too (the part is gone for good)\n'
    + 'Cancel = keep the file (the part is rebuilt on the next refresh)',
  );
  return deletePart(partId, source);
}

/**
 * Upload every chosen file in turn.  A name the server would reject is made
 * safe first; a file that already exists asks once before overwriting.
 */
export async function addFiles(files, report) {
  const results = [];
  for (const file of files) {
    const name = safeStepName(file.name);
    try {
      results.push(await uploadStep(file, name, false));
      report(`${name}: queued`);
    } catch (error) {
      if (error.conflict && window.confirm(`${name} already exists in the input directory. Replace it?`)) {
        try {
          results.push(await uploadStep(file, name, true));
          report(`${name}: replaced and queued`);
        } catch (again) {
          report(`${name}: ${again.message}`, true);
        }
      } else {
        report(`${name}: ${error.message}`, true);
      }
    }
  }
  return results;
}

// ----------------------------------------------------------------- job bar

/**
 * The strip under the header: what is running, its last log line, how many
 * jobs wait, and the outcome of the last finished job.  Hidden while there is
 * nothing to say.
 */
export function renderJobBar(element, status, note) {
  if (!element) return;
  const current = status && status.current;
  const queue = (status && status.queue) || [];
  const recent = (status && status.recent) || [];
  const last = recent[0];
  const parts = [];

  if (current) {
    const log = current.log || [];
    parts.push(`<span class="job-now"><span class="refresh-dot"></span>${escapeHtml(describeJob(current))}</span>`);
    if (log.length) parts.push(`<span class="job-log" title="${escapeHtml(log.slice(-8).join('\n'))}">${escapeHtml(log[log.length - 1])}</span>`);
  } else if (last && last.exitCode !== 0) {
    const log = last.log || [];
    parts.push(`<span class="job-failed">${escapeHtml(describeJob(last))} failed (exit ${last.exitCode})</span>`);
    // The line that names the failing part beats the trailing summary.
    const failLine = [...log].reverse().find((line) => /^FAIL\b/.test(line)) || log[log.length - 1];
    if (failLine) parts.push(`<span class="job-log" title="${escapeHtml(log.slice(-12).join('\n'))}">${escapeHtml(failLine)}</span>`);
  }
  if (queue.length) parts.push(`<span class="job-queue">${queue.length} queued</span>`);
  if (note) parts.push(`<span class="job-note${note.error ? ' error' : ''}">${escapeHtml(note.text)}</span>`);

  element.hidden = parts.length === 0;
  element.innerHTML = parts.join('');
}

