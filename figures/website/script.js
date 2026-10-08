'use strict';
const copyButton = document.getElementById('copy-citation');
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex').textContent.trim();
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation);
    status.textContent = 'Citation copied.';
    copyButton.textContent = 'Copied';
    setTimeout(() => { copyButton.textContent = 'Copy citation'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
