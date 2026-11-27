(() => {
  const THEME_KEY = 'driveaudit-theme';
  const LANG_KEY = 'driveaudit-lang';

  const SUN_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32 1.41-1.41"/></svg>';
  const MOON_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"/></svg>';

  const ZH = {
    'nav.overview': '概览',
    'nav.leaderboard': '榜单',
    'nav.comparisons': '可视化对比',
    'nav.closedLoop': '闭环测试',
    'nav.code': '代码',
    'cover.montage': '驾驶场景集锦',
    'cover.syncViews': '多视角同步',
    'cover.title': '超越视觉逼真度：面向驾驶策略测试的生成场景基准评估',
    'cover.explore': '浏览可视化对比',
    'cover.expandedHeading': '从视觉逼真度迈向驾驶策略测试',
    'cover.scroll': '向下滚动探索',
    'dim.obs.title': '观测质量',
    'dim.obs.desc': '即使底层交通布局合理，渲染模糊、结构畸变或帧间闪烁也会直接干扰自动驾驶策略的视觉感知。在将行为退化归因于策略缺陷之前，我们首先在全部六个同步相机视角上审计单帧感知质量与跨帧外观稳定性。',
    'dim.fid.title': '场景保真度',
    'dim.fid.desc': '真实记录只是未来演化的一种可能，因此生成场景不仅要画面逼真，更必须在三维几何与交通语义上保持自洽。我们重点检验相邻重叠视角间的跨视角极线几何一致性，以及渲染画面与评分所用交通轨迹之间的一致程度。',
    'dim.pol.title': '策略响应',
    'dim.pol.desc': '我们在涵盖白天、雨天、阴天与夜间的六个典型 nuScenes 场景中，将同一组六个冻结端到端驾驶策略运行于配对生成的多视角输入上。在下方选择样例并切换不同策略，可在鸟瞰图（BEV）中直观观察视觉观测差异如何改变规划轨迹，并进一步影响安全裕度、可行驶区域合规性与路线推进。',
    'dim.eff.title': '测试有效性',
    'overview.title': '基准概览',
    'overview.lede': '<strong>一段逼真的视频，并不等同于可信的驾驶测试。</strong>DriveAudit 并非仅从画面逼真度评判生成结果，而是从生成场景所诱发的自动驾驶开环与闭环策略行为出发进行系统评估。基准在严格对齐场景身份、时间戳、相机标定与自车运动的条件下配对对比真实录像与生成分支，贯通「观测质量、场景保真度、策略响应、质量感知测试有效性」四个耦合评估层级，从而准确区分真实的策略薄弱环节与由渲染伪影或不合理交通流引发的伪失效。',
    'lb.title': '排行榜结果',
    'lb.catk': '共享 CAT-K 交通流',
    'lb.recorded': '真实记录交通流',
    'lb.find': '搜索渲染模型',
    'lb.searchPlaceholder': '输入模型名称',
    'lb.rankBy': '排序指标',
    'lb.order': '排序方式',
    'lb.highToLow': '从高到低',
    'lb.lowToHigh': '从低到高',
    'lb.visibleMetrics': '显示指标',
    'lb.reset': '重置',
    'lb.captionCatk': '共享 CAT-K 交通轨迹 + 渲染',
    'lb.captionRecorded': '固定真实记录轨迹 + 渲染',
    'lb.rank': '排名',
    'lb.renderer': '渲染模型',
    'lb.route': '路线完成度',
    'lb.routeUp': '路线完成度 ↑',
    'lb.realVideo': '真实视频',
    'lb.reference': '参考基准',
    'lb.empty': '没有匹配该搜索条件的渲染模型，真实视频参考基准单独显示在底部。',
    'proto.title': '数据集与评估协议',
    'proto.lede': '七个主流渲染模型均在同一套 550 个多源历史场景、对齐的交通条件、统一评分基准以及固定六策略组合（UniAD、VAD、DrivoR、SparseDriveV2、DiffusionDriveV2、GoalFlow）下接受评估。通过固定历史输入与冻结驾驶策略，能够清晰解耦生成视觉观测与交通流对策略行为的因果影响。',
    'proto.col1Title': '同一共享历史，两种运动设定',
    'proto.col1P1': '从 1 秒（10 帧）的六视角历史观测出发，DriveAudit 在两种运动设定下对每个渲染模型进行未来 80 帧的对比评估：一是固定真实记录交通轨迹，二是共享 CAT-K 生成交通流。各冻结策略在对齐时间戳上预测规划轨迹；在主榜单的非响应式回放协议中，策略输出不会改变后续画面与交通演化。',
    'proto.col1P2': '评估接口严格限定为纯视觉（Camera-only），确保策略决策真实反映多视角生成图像的质量而非特权三维真值。对于架构上需要激光雷达输入张量的策略，按统一传感器协议传入零值/空激光雷达兼容输入。',
    'proto.col2Title': '五大公开数据源，来源清晰可溯',
    'proto.h250': '250 个历史片段',
    'proto.h100': '100 个历史片段',
    'proto.h75': '75 个历史片段',
    'proto.h50': '50 个历史片段',
    'comp.title': '可视化对比',
    'metric.clip.desc': 'CLIP-IQA+ 用于衡量无参考感知图像质量，重点捕捉画面锐度、道路与车辆细节纹理、光照一致性以及可见生成伪影。分数越高代表相机观测越清晰逼真；拖动分割线可在六视角下直观对比高分与退化渲染效果。',
    'metric.dino.desc': 'DINO-T 基于自监督视觉表征衡量相邻帧间的时序特征一致性。分数越高说明随着自车行驶，周边车辆、车道线与静态建筑能够保持稳定的外观与身份连贯性，而不会出现形变、跳变或明显闪烁。',
    'metric.epi.desc': 'Epi@3 计算相邻重叠相机视角间在 3 像素极线误差阈值下的特征匹配内点比例。分数越高表明车辆、路缘与车道线在跨视角拼接处能够保持准确的三维空间投影一致，避免目标错位、重影或几何断裂。',
    'metric.tvc.desc': 'TVC（轨迹-视觉一致性）用于检验生成视频中渲染出的周边车辆，是否与策略评分所依据的条件交通轨迹在位置、尺度与朝向上保持一致。其中真实参考分支采用原始记录运动，两个生成样例均以共享 CAT-K 交通流作为条件。',
    'cmp.goodBad': '优秀 / 较差',
    'cmp.recGood': '真实 / 优秀',
    'cmp.recBad': '真实 / 较差',
    'cmp.allThree': '三图并排',
    'cmp.recorded': '真实录像',
    'cmp.good': '优秀样例',
    'cmp.bad': '较差样例',
    'cam.fl': '左前',
    'cam.f': '正前',
    'cam.fr': '右前',
    'cam.bl': '左后',
    'cam.b': '正后',
    'cam.br': '右后',
    'case.1': '样例 1',
    'case.2': '样例 2',
    'case.3': '样例 3',
    'case.4': '样例 4',
    'case.5': '样例 5',
    'case.6': '样例 6',
    'scene.daylight': '白天',
    'scene.rain': '雨天',
    'scene.night': '夜间',
    'scene.sunny': '晴天',
    'scene.overcast': '阴天',
    'bev.opendwm': 'OpenDWM 规划',
    'bev.comparison': '对比规划',
    'bev.ego': '自车',
    'bev.vehicle': '其他车辆',
    'bev.pedestrian': '行人 / 骑行者',
    'loop.title': '闭环仿真研究',
    'loop.lede': '主榜单基于 550 个场景采用非响应式开环回放，而真实测试需要引入动作反馈闭环。本独立闭环研究基于 20 个配对场景与 CAT-K 交通流：每轮由驾驶策略依据渲染多视角图像生成规划，执行器沿规划前 0.5 秒推进自车位姿并同步更新周边交通状态，再由渲染器生成下一时刻观测。由于测试子集与闭环动力学不同，其绝对分数不应与主榜单直接横向对比。',
    'loop.s1': '滨水路段',
    'loop.s2': '日间右转',
    'loop.s3': '林荫街道',
    'loop.s4': '夜间路口',
    'loop.playTogether': '同步播放',
    'loop.simTime': '仿真时间',
    'footer.tagline': '超越视觉逼真度：面向驾驶策略测试的生成场景基准评估。',
    'footer.github': 'GitHub 代码仓库',
    'footer.sections': '页面导航',
    'footer.docs': '相关文档',
    'footer.docChallenge': '赛题与提交规范',
    'footer.docEval': '评估指南',
    'footer.docClosed': '闭环测试文档',
    'footer.backToTop': '回到顶部',
    'footer.topShort': '顶部',
    'footer.credit': 'DriveAudit 基准 · 自动驾驶世界模型配对评估框架',
    'footer.visitors': '访客数',
    'footer.views': '总浏览量',
  };

  const enText = new Map();
  const enHtml = new Map();
  const enPlaceholder = new Map();

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (!enText.has(key)) enText.set(key, el.textContent);
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const key = el.dataset.i18nHtml;
    if (!enHtml.has(key)) enHtml.set(key, el.innerHTML);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.dataset.i18nPlaceholder;
    if (!enPlaceholder.has(key)) enPlaceholder.set(key, el.getAttribute('placeholder') || '');
  });

  const themeBtn = document.querySelector('#theme-toggle');
  const langBtn = document.querySelector('#lang-toggle');
  const expandedWordmark = document.querySelector('.cover-expanded-wordmark');
  const footerWordmark = document.querySelector('.footer-wordmark');
  const backToTopBtn = document.querySelector('#back-to-top');
  const progressBar = document.querySelector('#scroll-progress');
  const navLinks = [...document.querySelectorAll('.cover-nav nav a[href^="#"]')];
  const trackedSections = navLinks
    .map(link => ({ link, el: document.querySelector(link.getAttribute('href')) }))
    .filter(item => item.el);

  let currentTheme = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
  let currentLang = 'en';
  try {
    const savedTheme = localStorage.getItem(THEME_KEY);
    if (savedTheme === 'dark' || savedTheme === 'light') currentTheme = savedTheme;
    const savedLang = localStorage.getItem(LANG_KEY);
    if (savedLang === 'zh' || savedLang === 'en') currentLang = savedLang;
  } catch {}

  function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.dataset.theme = theme;
    const wordmarkSrc = theme === 'dark'
      ? 'media/wordmark/driveaudit-on-dark.svg'
      : 'media/wordmark/driveaudit-on-light.svg';
    if (expandedWordmark) expandedWordmark.src = wordmarkSrc;
    if (footerWordmark) footerWordmark.src = wordmarkSrc;
    if (themeBtn) {
      const isDark = theme === 'dark';
      const label = currentLang === 'zh'
        ? (isDark ? '浅色' : '深色')
        : (isDark ? 'Light' : 'Dark');
      themeBtn.innerHTML = (isDark ? SUN_SVG : MOON_SVG) + '<span>' + label + '</span>';
      themeBtn.setAttribute('aria-pressed', String(isDark));
      themeBtn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    }
    try { localStorage.setItem(THEME_KEY, theme); } catch {}
  }

  function applyLang(lang) {
    currentLang = lang;
    const isZh = lang === 'zh';
    document.documentElement.lang = isZh ? 'zh-CN' : 'en';
    document.title = isZh ? 'DriveAudit | 超越视觉逼真度' : 'DriveAudit | Beyond Visual Realism';

    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.dataset.i18n;
      el.textContent = isZh ? (ZH[key] ?? enText.get(key) ?? el.textContent) : (enText.get(key) ?? el.textContent);
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.dataset.i18nHtml;
      el.innerHTML = isZh ? (ZH[key] ?? enHtml.get(key) ?? el.innerHTML) : (enHtml.get(key) ?? el.innerHTML);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.dataset.i18nPlaceholder;
      el.setAttribute('placeholder', isZh ? (ZH[key] ?? enPlaceholder.get(key) ?? '') : (enPlaceholder.get(key) ?? ''));
    });

    if (langBtn) {
      langBtn.textContent = isZh ? 'EN' : '中文';
      langBtn.setAttribute('aria-label', isZh ? 'Switch to English' : '切换为中文');
    }
    applyTheme(currentTheme);
    try { localStorage.setItem(LANG_KEY, lang); } catch {}
    dispatchEvent(new CustomEvent('driveaudit-lang-change', { detail: { lang } }));
  }

  if (themeBtn) {
    themeBtn.addEventListener('click', () => applyTheme(currentTheme === 'dark' ? 'light' : 'dark'));
  }
  if (langBtn) {
    langBtn.addEventListener('click', () => applyLang(currentLang === 'zh' ? 'en' : 'zh'));
  }

  document.querySelectorAll('[data-scroll-top]').forEach(btn => {
    btn.addEventListener('click', () => {
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' });
    });
  });

  function updateScrollUI() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
    const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    if (progressBar) {
      const ratio = Math.max(0, Math.min(1, scrollTop / maxScroll));
      progressBar.style.transform = `scaleX(${ratio.toFixed(4)})`;
    }
    if (backToTopBtn) {
      backToTopBtn.hidden = scrollTop < 480;
    }
    if (trackedSections.length) {
      const offset = 160;
      let activeLink = null;
      for (const { link, el } of trackedSections) {
        if (el.getBoundingClientRect().top <= offset) activeLink = link;
      }
      navLinks.forEach(link => link.classList.toggle('is-active', link === activeLink));
    }
  }

  let scrollTick = false;
  addEventListener('scroll', () => {
    if (scrollTick) return;
    scrollTick = true;
    requestAnimationFrame(() => {
      scrollTick = false;
      updateScrollUI();
    });
  }, { passive: true });
  addEventListener('resize', updateScrollUI, { passive: true });

  function initVisitorCounter() {
    const uvEl = document.getElementById('busuanzi_value_site_uv');
    const pvEl = document.getElementById('busuanzi_value_site_pv');
    if (!uvEl || !pvEl) return;

    const formatNumeric = el => {
      const raw = (el.textContent || '').trim();
      if (/^\d{4,}$/.test(raw)) el.textContent = Number(raw).toLocaleString();
    };

    const isLocal = location.protocol === 'file:' ||
      location.hostname === 'localhost' ||
      location.hostname === '127.0.0.1' ||
      location.hostname === '[::1]';

    if (isLocal) {
      try {
        const pv = (parseInt(localStorage.getItem('driveaudit-local-pv') || '0', 10) || 0) + 1;
        localStorage.setItem('driveaudit-local-pv', String(pv));
        let uv = parseInt(localStorage.getItem('driveaudit-local-uv') || '0', 10) || 0;
        if (!uv) {
          uv = 1;
          localStorage.setItem('driveaudit-local-uv', '1');
        }
        uvEl.textContent = uv.toLocaleString();
        pvEl.textContent = pv.toLocaleString();
      } catch {
        uvEl.textContent = '1';
        pvEl.textContent = '1';
      }
      return;
    }

    const hasValue = () => /\d/.test(uvEl.textContent || '') || /\d/.test(pvEl.textContent || '');
    const hasBothValues = () => /\d/.test(uvEl.textContent || '') && /\d/.test(pvEl.textContent || '');
    const providers = [
      'https://events.vercount.one/js',
      'https://cn.vercount.one/js',
      'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js',
    ];

    let idx = 0;
    let fallbackTimer = null;

    const observer = new MutationObserver(() => {
      if (hasValue()) {
        clearTimeout(fallbackTimer);
        formatNumeric(uvEl);
        formatNumeric(pvEl);
      }
      if (hasBothValues()) {
        observer.disconnect();
      }
    });
    observer.observe(uvEl, { childList: true, characterData: true, subtree: true });
    observer.observe(pvEl, { childList: true, characterData: true, subtree: true });

    function loadNext() {
      if (hasValue() || idx >= providers.length) return;
      const src = providers[idx++];
      const script = document.createElement('script');
      script.async = true;
      script.src = src;
      script.onerror = () => {
        clearTimeout(fallbackTimer);
        loadNext();
      };
      document.head.appendChild(script);
      clearTimeout(fallbackTimer);
      fallbackTimer = setTimeout(() => {
        if (!hasValue()) loadNext();
      }, 4500);
    }

    loadNext();
  }

  applyLang(currentLang);
  updateScrollUI();
  initVisitorCounter();
})();
