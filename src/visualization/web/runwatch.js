// "The run under this page moved on."
//
// Every page reads its artifacts out of one build.  The store publishes a new
// build file by file (store/sync.py::_publish, one os.replace each), so
// "old view.json + new mesh.json" is a reachable state for anything reading
// while a refresh lands.  Two halves guard against it:
//
//   * the requests say which build they want — data.js sends `If-Match` with
//     the run id this page loaded, and the server answers 409 rather than
//     handing over a file from a newer build (server.py::_if_match_satisfied);
//   * the page says so — this module, one banner with the same words on all
//     three pages, offering the reload rather than performing it.
//
// Nothing here is reloaded under the reviewer's feet.  A ranking or an edit
// path computed against the previous build is still the answer to the
// question that was asked; what the reader must not do is *mistake* it for
// the current one.

const POLL_MS = 5000;

/** The one sentence, so the three pages cannot drift apart in wording. */
export const RUN_MOVED_TEXT =
  'The run was rebuilt after this page loaded — what is on screen came from the previous build.';

/**
 * Draw (or hide) the "run moved on" banner in *element*.
 *
 * `shownRun` is the run id the page's data came from, `currentRun` the one on
 * disk now.  Unknown on either side means nothing is claimed: a page that
 * never learned its run id has nothing to compare, and saying so would be
 * noise.
 */
export function renderRunBanner(element, { shownRun, currentRun, onReload } = {}) {
  if (!element) return false;
  element.innerHTML = '';
  element.hidden = true;
  element.dataset.state = '';
  if (!shownRun || !currentRun || shownRun === currentRun) return false;
  element.dataset.state = 'updated';
  const chip = document.createElement('span');
  chip.className = 'st-chip';
  chip.textContent = 'rebuilt';
  const text = document.createElement('span');
  text.className = 'banner-text';
  text.textContent = RUN_MOVED_TEXT;
  element.append(chip, text);
  if (onReload) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn banner-btn';
    button.textContent = 'Reload page';
    button.addEventListener('click', onReload);
    element.append(button);
  }
  element.hidden = false;
  return true;
}

/**
 * Ask `/api/status` for the run id on disk, forever, and hand each answer to
 * *onRunId*.  Returns the stop function.
 *
 * Only for pages that do not already poll: the part page and the launcher
 * both read `dataRunId` out of a poll they run anyway, and a second timer
 * over the same endpoint would just double the traffic.
 *
 * This one keeps polling on a read-only server (`--no-refresh-api`), where
 * the other two stop: nothing the *page* does can change the run there, but
 * a `refresh` in a terminal still can, and that is the usual way of working
 * with a read-only viewer.
 */
export function pollRunId(onRunId, interval = POLL_MS) {
  let timer = null;
  let stopped = false;
  const tick = async () => {
    let runId = null;
    try {
      const status = await fetch('api/status', { cache: 'no-store' }).then((r) => r.json());
      runId = (status && status.dataRunId) || null;
    } catch {
      runId = null;
    }
    if (stopped) return;
    if (runId) {
      try { onRunId(runId); } catch (error) { console.error(error); }
    }
    timer = setTimeout(tick, interval);
  };
  tick();
  return () => { stopped = true; clearTimeout(timer); };
}

/**
 * A full re-take of the page.
 *
 * Plain `reload()` is enough here: run data is served `no-store` and the
 * page's own modules revalidate on every request (server.py, `no-cache` plus
 * a content ETag), so nothing stale can survive it.
 */
export function reloadPage() {
  location.reload();
}
