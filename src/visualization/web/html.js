// The one HTML escaper.
//
// Most of the pages build their DOM node by node, where the browser does this
// for us.  The handful of places that assemble a small string of markup are
// the exception, and they interpolate values the page did not write: relation
// type names and breakdown channel keys out of the artifacts, build times,
// and the message of whatever error just came back.  None of those may
// contain markup today - the vocabularies are Python identifiers and the
// timestamps are written by the pipeline - but standing surety for the
// content of somebody else's data is not the page's job, and a stray `<`
// there is the difference between an odd-looking word and a broken layout.
//
// It lived in two copies, character for character, in parts.js and
// compare.js, and half the sinks used neither.

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]
  ));
}
