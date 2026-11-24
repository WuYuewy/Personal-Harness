/* Chainlit 2.12 MultiSelect labels point to an absent input ID.
 * Keep its native generated IDs and add an accessible name for the actual input. */
(() => {
  function nameMultiSelects() {
    for (const label of document.querySelectorAll('label[for="tools"], label[for="tool_permissions"]')) {
      const input = label.parentElement?.querySelector('input[role="combobox"]');
      if (!input) continue;
      const identity = `harness-label-${label.htmlFor}`;
      if (label.id !== identity) label.id = identity;
      if (input.getAttribute("aria-labelledby") !== identity) input.setAttribute("aria-labelledby", identity);
    }
  }
  nameMultiSelects();
  new MutationObserver(nameMultiSelects).observe(document.documentElement, {childList: true, subtree: true, attributes: true, attributeFilter: ["aria-labelledby"]});
})();
