(() => {
  const root = document.querySelector('#leaderboard');
  if (!root) return;
  const controls = root.querySelector('.leaderboard-controls');
  const status = root.querySelector('.leaderboard-status');
  const search = root.querySelector('#renderer-search');
  const metricSelect = root.querySelector('#ranking-metric');
  const directionSelect = root.querySelector('#ranking-direction');
  const reset = root.querySelector('#leaderboard-reset');
  const picker = root.querySelector('.column-picker');
  const settings = [...root.querySelectorAll('[data-setting-target]')];
  const toggles = [...root.querySelectorAll('[data-metric-toggle]')];
  const panels = [...root.querySelectorAll('[data-results-setting]')];
  if (!controls || !search || !metricSelect || !directionSelect || !reset || !picker || panels.length !== 2) return;

  // The two semantic HTML tables are the only score source. With no JavaScript,
  // both remain readable and the enhancement controls remain hidden.
  const metrics = new Map([...panels[0].querySelectorAll('thead [data-metric]')].map(header => [
    header.dataset.metric,
    { label: header.dataset.label, direction: header.dataset.direction },
  ]));
  const tables = panels.map(panel => {
    const table = panel.querySelector('table');
    const rows = [...table.tBodies[0].rows].map((element, index) => {
      const scores = new Map([...element.querySelectorAll('td[data-metric]')].map(cell => [
        cell.dataset.metric, cell.textContent.trim() === '' ? NaN : Number(cell.textContent.trim()),
      ]));
      if (scores.size !== metrics.size || [...scores.values()].some(value => !Number.isFinite(value))) {
        throw new Error('Incomplete reported leaderboard row; keep the static source tables intact.');
      }
      return { element, index, name: element.dataset.renderer, scores };
    });
    return { panel, table, rows, mode: panel.dataset.resultsSetting, empty: panel.querySelector('.leaderboard-empty') };
  });
  const state = { mode: 'catk', metric: 'qtes', direction: 'desc', query: '', visible: new Set(metrics.keys()) };
  const normalize = value => value.toLowerCase().replace(/[\s_-]+/g, '');
  let message = '';

  function render() {
    const factor = state.direction === 'asc' ? 1 : -1;
    let shown = 0;
    let total = 0;
    for (const entry of tables) {
      entry.panel.hidden = entry.mode !== state.mode;
      const sorted = [...entry.rows].sort((a, b) => factor * (a.scores.get(state.metric) - b.scores.get(state.metric)) || a.index - b.index);
      let previous = null;
      let rank = 0;
      let matches = 0;
      sorted.forEach((row, index) => {
        const value = row.scores.get(state.metric);
        if (previous === null || value !== previous) rank = index + 1;
        previous = value;
        row.element.querySelector('.rank-value').textContent = rank;
        row.element.dataset.rank = String(rank);
        row.element.hidden = !normalize(row.name).includes(state.query);
        if (!row.element.hidden) matches += 1;
        entry.table.tBodies[0].append(row.element);
      });
      entry.empty.hidden = matches !== 0;
      entry.table.style.minWidth = `${Math.max(400, 208 + state.visible.size * 106)}px`;
      entry.table.querySelector('.rank-heading').setAttribute('aria-label', `Rank by ${metrics.get(state.metric).label}`);
      entry.table.querySelectorAll('[data-metric]').forEach(cell => {
        const key = cell.dataset.metric;
        cell.hidden = !state.visible.has(key);
        cell.classList.toggle('is-sort-column', key === state.metric);
        if (cell.parentElement.parentElement.tagName === 'THEAD') {
          cell.removeAttribute('aria-sort');
          if (key === state.metric) cell.setAttribute('aria-sort', state.direction === 'asc' ? 'ascending' : 'descending');
        }
      });
      if (entry.mode === state.mode) { shown = matches; total = sorted.length; }
    }
    settings.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.settingTarget === state.mode)));
    for (const option of metricSelect.options) option.disabled = !state.visible.has(option.value);
    metricSelect.value = state.metric;
    directionSelect.value = state.direction;
    root.querySelectorAll('[data-rank-by]').forEach(button => {
      const key = button.dataset.rankBy;
      const metric = metrics.get(key);
      const meaning = metric.direction === 'asc' ? 'Lower is better.' : 'Higher is better.';
      const current = key === state.metric ? ` Currently ${state.direction === 'asc' ? 'low to high' : 'high to low'}; activate to reverse.` : '';
      button.setAttribute('aria-label', `Rank by ${metric.label}. ${meaning}${current}`);
    });
    if (status) {
      const label = state.mode === 'catk' ? 'Shared CAT-K traffic' : 'Recorded traffic';
      status.textContent = `${shown} of ${total} renderers · ${label} · ${metrics.get(state.metric).label}, ${state.direction === 'asc' ? 'low to high' : 'high to low'}.${message ? ' ' + message : ''}`;
    }
    message = '';
  }

  settings.forEach(button => button.addEventListener('click', () => {
    state.mode = button.dataset.settingTarget;
    render();
  }));
  search.addEventListener('input', () => { state.query = normalize(search.value.trim()); render(); });
  metricSelect.addEventListener('change', () => {
    if (!state.visible.has(metricSelect.value)) return;
    state.metric = metricSelect.value;
    state.direction = metrics.get(state.metric).direction;
    render();
  });
  directionSelect.addEventListener('change', () => { state.direction = directionSelect.value; render(); });
  root.querySelectorAll('[data-rank-by]').forEach(button => button.addEventListener('click', () => {
    const key = button.dataset.rankBy;
    if (key === state.metric) state.direction = state.direction === 'asc' ? 'desc' : 'asc';
    else { state.metric = key; state.direction = metrics.get(key).direction; }
    render();
  }));
  toggles.forEach(toggle => toggle.addEventListener('change', () => {
    const key = toggle.dataset.metricToggle;
    if (toggle.checked) state.visible.add(key);
    else if (state.visible.size === 1) {
      toggle.checked = true;
      message = 'Keep at least one metric visible.';
    } else state.visible.delete(key);
    if (!state.visible.has(state.metric)) {
      state.metric = state.visible.has('qtes') ? 'qtes' : [...metrics.keys()].find(key => state.visible.has(key));
      state.direction = metrics.get(state.metric).direction;
    }
    render();
  }));
  reset.addEventListener('click', () => {
    state.mode = 'catk'; state.metric = 'qtes'; state.direction = 'desc'; state.query = '';
    state.visible = new Set(metrics.keys());
    search.value = '';
    toggles.forEach(toggle => { toggle.checked = true; });
    picker.open = false;
    render();
  });
  document.addEventListener('pointerdown', event => {
    if (picker.open && !picker.contains(event.target)) picker.open = false;
  });
  root.addEventListener('keydown', event => {
    if (event.key === 'Escape' && picker.open) {
      picker.open = false;
      picker.querySelector('summary').focus();
    }
  });

  render();
  root.querySelectorAll('.metric-static').forEach(label => { label.hidden = true; });
  root.querySelectorAll('.table-sort').forEach(button => { button.hidden = false; });
  if (status) status.hidden = false;
  controls.hidden = false;
})();
