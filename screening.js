const demos = {
  "d1-1": [
    "BaseDreamer",
    "MagicDrive-V2",
    1,
    "Front left"
  ],
  "d1-2": [
    "HorizonDrive",
    "MagicDrive-V2",
    2,
    "Front"
  ],
  "d1-3": [
    "Panacea",
    "MagicDrive-V2",
    1,
    "Front right"
  ],
  "d1-4": [
    "DreamForge",
    "MagicDrive-V2",
    2,
    "Back left"
  ],
  "d1-5": [
    "BaseDreamer",
    "MagicDrive",
    1,
    "Back"
  ],
  "d1-6": [
    "BaseDreamer",
    "OpenDWM",
    2,
    "Back right"
  ],
  "d2-1": [
    "OpenDWM",
    "MagicDrive",
    1,
    "Front"
  ],
  "d2-2": [
    "OpenDWM",
    "Panacea",
    2,
    "Front right"
  ],
  "d2-3": [
    "HorizonDrive",
    "MagicDrive",
    1,
    "Back left"
  ],
  "d2-4": [
    "HorizonDrive",
    "Panacea",
    2,
    "Back"
  ],
  "d2-5": [
    "OpenDWM",
    "BaseDreamer",
    1,
    "Back right"
  ],
  "d2-6": [
    "DreamForge",
    "MagicDrive",
    2,
    "Front left"
  ],
  "d3-1": [
    "DreamForge",
    "MagicDrive",
    2,
    "Front right"
  ],
  "d3-2": [
    "HorizonDrive",
    "MagicDrive",
    1,
    "Back left"
  ],
  "d3-3": [
    "OpenDWM",
    "MagicDrive",
    2,
    "Back"
  ],
  "d3-4": [
    "BaseDreamer",
    "MagicDrive",
    1,
    "Back right"
  ],
  "d3-5": [
    "MagicDrive-V2",
    "MagicDrive",
    2,
    "Front left"
  ],
  "d3-6": [
    "Panacea",
    "MagicDrive",
    1,
    "Front"
  ],
  "d4-1": [
    "OpenDWM",
    "MagicDrive",
    2,
    "Back left"
  ],
  "d4-2": [
    "OpenDWM",
    "BaseDreamer",
    1,
    "Back"
  ],
  "d4-3": [
    "OpenDWM",
    "Panacea",
    2,
    "Back right"
  ],
  "d4-4": [
    "OpenDWM",
    "MagicDrive-V2",
    1,
    "Front left"
  ],
  "d4-5": [
    "DreamForge",
    "MagicDrive",
    2,
    "Front"
  ],
  "d4-6": [
    "OpenDWM",
    "HorizonDrive",
    1,
    "Front right"
  ]
};
const benchmark = {
  "OpenDWM": [
    ".531",
    ".730",
    ".222",
    ".543",
    ".724",
    ".878",
    "44.079"
  ],
  "MagicDrive-V2": [
    ".437",
    ".744",
    ".179",
    ".447",
    ".718",
    ".870",
    "39.509"
  ],
  "Panacea": [
    ".581",
    ".728",
    ".164",
    ".374",
    ".717",
    ".881",
    "39.146"
  ],
  "DreamForge": [
    ".573",
    ".705",
    ".168",
    ".495",
    ".729",
    ".881",
    "41.311"
  ],
  "BaseDreamer": [
    ".661",
    ".677",
    ".143",
    ".410",
    ".721",
    ".880",
    "38.522"
  ],
  "HorizonDrive": [
    ".593",
    ".842",
    ".116",
    ".519",
    ".725",
    ".880",
    "40.108"
  ],
  "MagicDrive": [
    ".530",
    ".681",
    ".132",
    ".363",
    ".670",
    ".855",
    "37.282"
  ]
};
const metricNames = ["CLIP-IQA+", "DINO-T", "Epi@3", "TVC", "RPDMS", "Route completion", "QTES"];
const dimensions = [[0,1],[2,3],[4,5],[6]];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const cover = document.querySelector('#cover-film');
const coverToggle = document.querySelector('#cover-toggle');
coverToggle.addEventListener('click', () => {
  if (cover.paused) cover.play().catch(() => {});
  else cover.pause();
});
cover.addEventListener('play', () => { coverToggle.textContent = 'Pause cover'; });
cover.addEventListener('pause', () => { coverToggle.textContent = 'Play cover'; });
if (!reduceMotion.matches) cover.play().catch(() => {});
document.querySelectorAll('.dimension').forEach((section, index) => {
  const video = section.querySelector('video');
  const buttons = section.querySelectorAll('[data-demo]');
  buttons.forEach(button => button.addEventListener('click', () => {
    const key = button.dataset.demo;
    const [left, right, scene, camera] = demos[key];
    buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    section.querySelectorAll('.pairlabels span').forEach((label, i) => { label.textContent = [left, right][i]; });
    section.querySelectorAll('thead th').forEach((cell, i) => { if (i) cell.textContent = [left, right][i-1]; });
    section.querySelectorAll('tbody tr').forEach((row, i) => {
      const metric = dimensions[index][i];
      row.querySelectorAll('td').forEach((cell, j) => { cell.textContent = Number(benchmark[[left,right][j]][metric]).toFixed(3); });
    });
    section.querySelector('figcaption').textContent = 'nuScenes case_' + String(scene).padStart(6,'0') + ' / ' + camera + ' / 80 future frames';
    video.poster = 'media/atlas/' + key + '.jpg';
    video.src = 'media/atlas/' + key + '.mp4';
    video.play().catch(() => {});
  }));
  video.addEventListener('play', () => {
    document.querySelectorAll('.dimension video, #cover-film').forEach(other => { if (other !== video) other.pause(); });
  });
});
