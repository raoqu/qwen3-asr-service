/* 实时转写页（Vue 3 + Naive UI，无构建 UMD）。
 * 协议：WS /v2/asr/stream（?token= 鉴权），start/binary/stop 信封；
 * 麦克风经 AudioWorklet（/web-ui/assets/pcm-worklet.js）采集，主线程重采样 16k Int16；
 * 文件模拟推流优先 ffmpeg-wasm（CDN 懒加载，失败回退浏览器原生解码），200ms 分帧 + 1MB 背压。
 */
(function () {
  'use strict';
  const { createApp, ref, reactive, computed, onMounted, onBeforeUnmount } = Vue;
  const { fmtMs, makeRoot } = window.AsrCommon;

  const RT_SR = 16000;
  const FRAME = 3200;                 // 200ms @16k
  const BP_LIMIT = 1 << 20;           // 1MB 发送缓冲上限（背压）
  const MAX_LOG_LINES = 300;
  const MAX_TRANSCRIPT_LINES = 200;
  const FFMPEG_MIRROR = 'https://unpkg.zhimg.com';
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function floatToInt16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const v = Math.max(-1, Math.min(1, f32[i]));
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    return out;
  }
  // 麦克风：任意采样率线性重采样到 16k 并转 Int16
  function micFloatTo16kPCM(input, inputSr) {
    if (inputSr === RT_SR) return floatToInt16(input);
    const ratio = inputSr / RT_SR, outLen = Math.floor(input.length / ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = i * ratio, i0 = Math.floor(idx), frac = idx - i0;
      const s = (input[i0] || 0) * (1 - frac) + (input[i0 + 1] || 0) * frac;
      const v = Math.max(-1, Math.min(1, s));
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    return out;
  }

  const AppBody = {
    props: { themeMode: { type: String, required: true } },
    emits: ['cycle-theme'],
    setup(props, { emit }) {
      // —— 连接配置（API Key 与离线页共用 localStorage 键）——
      const token = ref(localStorage.getItem('asr_api_key') || '');
      Vue.watch(token, v => localStorage.setItem('asr_api_key', v.trim()));
      const lang = ref('');

      // —— 会话状态机：idle | connecting | streaming | stopping ——
      const streamState = ref('idle');
      const statusText = ref('未连接');
      const statusType = computed(() => (
        streamState.value === 'streaming' ? 'error'                       // 红点=录制/推流中
          : streamState.value === 'idle' ? 'default' : 'warning'
      ));
      const source = ref('mic');          // mic | file
      const busy = computed(() => streamState.value !== 'idle');

      // —— 提示与能力 ——
      const capWarning = ref('');
      const hint = ref('');
      const capInfo = ref('');

      // —— 结果 ——
      const finals = reactive([]);        // {key, start, text, words}
      const partial = ref('');
      let finalSeq = 0;
      function appendFinal(m) {
        finals.push({ key: ++finalSeq, start: m.start, text: m.text || '', words: (m.words && m.words.length) || 0 });
        if (finals.length > MAX_TRANSCRIPT_LINES) finals.shift();
      }
      function clearResults() { finals.length = 0; partial.value = ''; }

      // —— 协议日志 ——
      const logs = reactive([]);          // {key, ts, kind, text}
      const logOpen = ref(false);
      let logSeq = 0;
      const logRef = ref(null);
      function log(kind, data) {
        const ts = new Date().toISOString().substr(11, 12);
        const body = typeof data === 'string' ? data : JSON.stringify(data);
        logs.push({ key: ++logSeq, ts, kind, text: body });
        if (logs.length > MAX_LOG_LINES) logs.shift();
        Vue.nextTick(() => { const el = logRef.value; if (el) el.scrollTop = el.scrollHeight; });
      }

      // —— 诊断（发送/接收速率、WS 缓冲、最大帧、主线程卡顿）——
      const diagText = ref('');
      let diagTimer = null, rafId = null, sentFrames = 0, recvMsgs = 0, lastRaf = 0, maxStall = 0, maxFrame = 0;
      function startDiag() {
        sentFrames = 0; recvMsgs = 0; maxStall = 0; lastRaf = 0; maxFrame = 0;
        const tick = now => {
          if (lastRaf) { const gap = now - lastRaf; if (gap > maxStall) maxStall = gap; }
          lastRaf = now;
          rafId = requestAnimationFrame(tick);
        };
        rafId = requestAnimationFrame(tick);
        diagTimer = setInterval(() => {
          const buf = ws ? ws.bufferedAmount : 0;
          diagText.value = '诊断 · 发送 ' + sentFrames + ' 帧/s · 接收 ' + recvMsgs + ' 条/s · WS缓冲 ' +
            Math.round(buf / 1024) + ' KB · 最大帧 ' + maxFrame + ' 样本 · 主线程最大卡顿 ' + Math.round(maxStall) + ' ms';
          sentFrames = 0; recvMsgs = 0; maxStall = 0; maxFrame = 0;
        }, 1000);
      }
      function stopDiag() {
        if (diagTimer) { clearInterval(diagTimer); diagTimer = null; }
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      }

      // —— WebSocket ——
      let ws = null;
      function wsSendJson(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(obj)); log('send', obj); }
      }
      function startMsg() {
        const l = lang.value.trim();
        const m = { type: 'start', audio_fs: RT_SR, wav_name: 'web-test' };
        if (l && l !== 'auto') m.language = l;
        return m;
      }
      function waitDrain() {
        return new Promise(res => {
          if (!ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount < BP_LIMIT) return res();
          const iv = setInterval(() => {
            if (!ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount < BP_LIMIT) { clearInterval(iv); res(); }
          }, 20);
        });
      }
      function openWs(onReady) {
        hint.value = '';
        clearResults();
        const t = token.value.trim();
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const url = proto + '://' + location.host + '/v2/asr/stream' + (t ? '?token=' + encodeURIComponent(t) : '');
        streamState.value = 'connecting';
        statusText.value = '连接中...';
        ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => log('evt', 'WS open');
        ws.onmessage = ev => {
          recvMsgs++;
          let m;
          try { m = JSON.parse(ev.data); } catch (e) { return; }
          log('recv', m);
          if (m.type === 'session.created') {
            capInfo.value = '协议 ' + m.protocol + ' v' + m.protocol_version + ' · 模式 ' + m.mode +
              ' · 后端 ' + m.backend + ' · 采样率 ' + m.sample_rate + ' · 能力 ' + JSON.stringify(m.capabilities);
            statusText.value = '已连接 · ' + m.backend;
            streamState.value = 'streaming';
            if (onReady) onReady();
          } else if (m.type === 'partial') {
            partial.value = m.text || '';
          } else if (m.type === 'final') {
            appendFinal(m);
            partial.value = '';
          } else if (m.type === 'error') {
            hint.value = '[' + m.code + '] ' + m.message;
          } else if (m.type === 'session.closed') {
            statusText.value = '会话结束';
          }
        };
        ws.onerror = () => log('evt', 'WS error');
        ws.onclose = ev => {
          log('evt', 'WS close code=' + ev.code);
          statusText.value = '已断开 (' + ev.code + ')';
          if (ev.code === 1008) hint.value = '鉴权失败：请检查 API Key。';
          else if (ev.code === 1013) hint.value = '并发会话已满（1013）。';
          else if (ev.code === 1011) hint.value = '实时端点未就绪：请用 --serve-mode standard --enable-stream 启动服务。';
          streamState.value = 'idle';
          cleanupMic();
        };
      }
      function closeWs() { try { if (ws) ws.close(); } catch (e) { /* 已断开 */ } }

      // —— 麦克风（AudioWorklet 外置文件）——
      let micCtx = null, micNode = null, micSrc = null, micStream = null;
      async function startMic() {
        hint.value = '';
        try {
          micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
          hint.value = '麦克风访问失败: ' + e.message;
          return;
        }
        openWs(async () => {
          wsSendJson(startMsg());
          // 直接建 16kHz 上下文：免主线程重采样、降低 worklet 投递频率；不支持则回退默认采样率
          const AC = window.AudioContext || window.webkitAudioContext;
          try { micCtx = new AC({ sampleRate: RT_SR }); } catch (e) { micCtx = new AC(); }
          try {
            await micCtx.audioWorklet.addModule('/web-ui/assets/pcm-worklet.js');
          } catch (e) {
            hint.value = 'AudioWorklet 加载失败: ' + e.message;
            cleanupMic(); closeWs();
            return;
          }
          micSrc = micCtx.createMediaStreamSource(micStream);
          micNode = new AudioWorkletNode(micCtx, 'pcm-worklet');
          micNode.port.onmessage = ev => {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            if (ev.data.length > maxFrame) maxFrame = ev.data.length;
            if (ws.bufferedAmount > BP_LIMIT) return;   // 背压丢帧
            ws.send(micFloatTo16kPCM(ev.data, micCtx.sampleRate).buffer);
            sentFrames++;
          };
          // worklet 不写 output → 输出静音；连 destination 仅为驱动音频图
          micSrc.connect(micNode);
          micNode.connect(micCtx.destination);
          startDiag();
          statusText.value = '🔴 录音中...';
        });
      }
      function stopMic() {
        streamState.value = 'stopping';
        wsSendJson({ type: 'stop' });
        cleanupMic();
        statusText.value = '已停止，等待末段结果...';
      }
      function cleanupMic() {
        stopDiag();
        try { if (micNode) { micNode.port.onmessage = null; micNode.disconnect(); micNode = null; } } catch (e) { /* noop */ }
        try { if (micSrc) { micSrc.disconnect(); micSrc = null; } } catch (e) { /* noop */ }
        try { if (micCtx) { micCtx.close(); micCtx = null; } } catch (e) { /* noop */ }
        try { if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; } } catch (e) { /* noop */ }
      }

      // —— ffmpeg-wasm 懒加载（外网 CDN；失败回退浏览器原生解码）——
      let ffmpeg = null, ffmpegTried = false;
      function loadScript(src) {
        return new Promise((res, rej) => {
          const s = document.createElement('script');
          s.src = src; s.async = true;
          s.onload = res;
          s.onerror = () => rej(new Error('脚本加载失败: ' + src));
          document.head.appendChild(s);
        });
      }
      async function getFFmpeg() {
        if (ffmpeg) return ffmpeg;
        if (ffmpegTried) return null;
        ffmpegTried = true;
        try {
          statusText.value = '正在加载转码器 (ffmpeg-wasm)…';
          if (!window.FFmpegWASM) await loadScript(FFMPEG_MIRROR + '/@ffmpeg/ffmpeg@0.12.10/dist/umd/ffmpeg.js');
          if (!window.FFmpegUtil) await loadScript(FFMPEG_MIRROR + '/@ffmpeg/util@0.12.1/dist/umd/index.js');
          const { FFmpeg } = window.FFmpegWASM;
          const { toBlobURL } = window.FFmpegUtil;
          const core = FFMPEG_MIRROR + '/@ffmpeg/core@0.12.10/dist/umd';
          const ff = new FFmpeg();
          ff.on('log', ({ message }) => log('evt', 'ffmpeg: ' + message));
          await ff.load({
            coreURL: await toBlobURL(core + '/ffmpeg-core.js', 'text/javascript'),
            wasmURL: await toBlobURL(core + '/ffmpeg-core.wasm', 'application/wasm'),
          });
          ffmpeg = ff;
          log('evt', 'ffmpeg-wasm 已就绪');
          return ff;
        } catch (e) {
          log('evt', 'ffmpeg-wasm 加载失败，回退浏览器原生解码: ' + e.message);
          return null;
        }
      }
      // 文件 → 16k 单声道 PCM16。优先 ffmpeg-wasm（直出 s16le），失败回退 Web Audio
      async function decodeFileTo16kPCM(file) {
        const ff = await getFFmpeg();
        if (ff) {
          const { fetchFile } = window.FFmpegUtil;
          await ff.writeFile('input', await fetchFile(file));
          await ff.exec(['-i', 'input', '-ac', '1', '-ar', String(RT_SR), '-f', 's16le', 'out.pcm']);
          const out = await ff.readFile('out.pcm');
          try { await ff.deleteFile('input'); await ff.deleteFile('out.pcm'); } catch (e) { /* noop */ }
          return new Int16Array(out.buffer, out.byteOffset, Math.floor(out.byteLength / 2));
        }
        const buf = await file.arrayBuffer();
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await ac.decodeAudioData(buf);
        ac.close();
        const dstLen = Math.ceil(decoded.length * RT_SR / decoded.sampleRate);
        const off = new OfflineAudioContext(1, dstLen, RT_SR);
        const bs = off.createBufferSource();
        bs.buffer = decoded; bs.connect(off.destination); bs.start();
        const rendered = await off.startRendering();
        return floatToInt16(rendered.getChannelData(0));
      }

      // —— 文件模拟推流 ——
      const fileInputRef = ref(null);
      const fileChosen = ref('');
      const noThrottle = ref(false);
      const fileProgress = ref(0);
      const fileRunning = ref(false);
      let fileAborted = false;
      function onFileChange() {
        const el = fileInputRef.value;
        fileChosen.value = el && el.files.length ? el.files[0].name : '';
      }
      async function startFile() {
        hint.value = '';
        const el = fileInputRef.value;
        const file = el && el.files[0];
        if (!file) { hint.value = '请先选择音频文件。'; return; }
        fileRunning.value = true;
        fileProgress.value = 0;
        fileAborted = false;

        let pcm16;
        try {
          pcm16 = await decodeFileTo16kPCM(file);
        } catch (e) {
          hint.value = '音频解码失败: ' + e.message;
          fileRunning.value = false;
          statusText.value = '未连接';
          return;
        }
        if (fileAborted) { fileRunning.value = false; return; }

        openWs(async () => {
          wsSendJson(startMsg());
          statusText.value = '📤 推流中...';
          startDiag();
          const total = pcm16.length;
          const frameMs = FRAME / RT_SR * 1000;   // 200ms
          for (let i = 0; i < total; i += FRAME) {
            if (fileAborted || !ws || ws.readyState !== WebSocket.OPEN) { fileRunning.value = false; return; }
            await waitDrain();                     // 背压：发送缓冲过高时等排空
            ws.send(pcm16.subarray(i, Math.min(i + FRAME, total)));
            sentFrames++;
            fileProgress.value = Math.round(Math.min(i + FRAME, total) / total * 100);
            if (!noThrottle.value) await sleep(frameMs);
          }
          if (fileAborted || !ws || ws.readyState !== WebSocket.OPEN) { fileRunning.value = false; return; }
          fileProgress.value = 100;
          streamState.value = 'stopping';
          wsSendJson({ type: 'stop' });
          statusText.value = '推流完成，等待末段结果...';
          fileRunning.value = false;
        });
      }
      function stopFile() {
        fileAborted = true;
        streamState.value = 'stopping';
        wsSendJson({ type: 'stop' });
        statusText.value = '已停止';
        fileRunning.value = false;
      }

      // —— 能力预检 ——
      async function precheck() {
        try {
          const r = await fetch('/v2/capabilities');
          if (!r.ok) return;
          const c = await r.json();
          if (!c.stream || !c.stream.enabled) {
            capWarning.value = '当前服务未启用实时端点。请用 --serve-mode standard --enable-stream 启动后刷新本页。';
          } else {
            capWarning.value = '';
          }
        } catch (e) { /* 服务未起，忽略 */ }
      }
      onMounted(precheck);
      onBeforeUnmount(() => { cleanupMic(); closeWs(); stopDiag(); });

      const themeLabel = computed(() => ({ auto: '🌗 跟随系统', light: '☀️ 浅色', dark: '🌙 深色' }[props.themeMode]));

      return {
        token, lang, themeLabel,
        streamState, statusText, statusType, busy, source,
        capWarning, hint, capInfo, diagText,
        finals, partial, fmtMs,
        logs, logOpen, logRef,
        fileInputRef, fileChosen, noThrottle, fileProgress, fileRunning, onFileChange,
        startMic, stopMic, startFile, stopFile,
        cycleTheme: () => emit('cycle-theme'),
      };
    },
    template: `
      <div>
        <n-alert v-if="capWarning" type="warning" :show-icon="true" style="margin-bottom:12px;">{{ capWarning }}</n-alert>

        <n-card title="⚙️ 连接" size="small" style="margin-bottom:14px;">
          <template #header-extra>
            <n-button size="small" quaternary @click="cycleTheme">{{ themeLabel }}</n-button>
          </template>
          <n-space align="center" :wrap="true">
            <n-input v-model:value="token" type="password" show-password-on="click" placeholder="API Key（留空表示无需认证）" size="small" style="width:260px;"></n-input>
            <n-input v-model:value="lang" placeholder="语言（默认 auto）" size="small" style="width:140px;"></n-input>
            <n-tag :type="statusType" :bordered="false" size="small">{{ statusText }}</n-tag>
          </n-space>
          <n-text v-if="diagText" depth="3" style="display:block;margin-top:8px;font-size:0.78em;">{{ diagText }}</n-text>
          <n-text v-if="capInfo" depth="3" style="display:block;margin-top:4px;font-size:0.78em;">{{ capInfo }}</n-text>
        </n-card>

        <n-card size="small" style="margin-bottom:14px;">
          <n-tabs v-model:value="source" type="segment" size="small">
            <n-tab-pane name="mic" tab="🎤 麦克风" :disabled="busy && source !== 'mic'">
              <n-space align="center" style="margin-top:8px;">
                <n-button v-if="!busy" id="micStart" type="primary" @click="startMic">开始录音</n-button>
                <n-button v-else type="error" :disabled="streamState === 'stopping'" @click="stopMic">停止</n-button>
                <n-text depth="3" style="font-size:0.85em;">点击后授权麦克风，边说边转写。</n-text>
              </n-space>
            </n-tab-pane>
            <n-tab-pane name="file" tab="📁 文件模拟" :disabled="busy && source !== 'file'">
              <n-space vertical style="margin-top:8px;">
                <input id="fileInput" ref="fileInputRef" type="file" accept="audio/*" @change="onFileChange">
                <n-space align="center">
                  <n-button v-if="!busy && !fileRunning" type="primary" @click="startFile">开始模拟推流</n-button>
                  <n-button v-else type="error" :disabled="streamState === 'stopping'" @click="stopFile">停止</n-button>
                  <n-checkbox v-model:checked="noThrottle">⚡ 不限速（尽快推送）</n-checkbox>
                </n-space>
                <n-text depth="3" style="font-size:0.8em;">
                  点击后按需加载 ffmpeg-wasm 解码（需外网，首次约 25–30MB，仅本次会话加载一次；加载失败自动回退浏览器原生解码）→ 转 16k 单声道 → 按 200ms 分帧推流，模拟实时输入。
                </n-text>
                <n-progress v-if="fileRunning || fileProgress > 0" type="line" :percentage="fileProgress" :indicator-placement="'inside'"></n-progress>
              </n-space>
            </n-tab-pane>
          </n-tabs>
          <n-alert v-if="hint" type="error" :show-icon="true" style="margin-top:10px;">{{ hint }}</n-alert>
        </n-card>

        <n-card title="📝 转写结果" size="small" style="margin-bottom:14px;">
          <div id="transcript">
            <n-empty v-if="!finals.length && !partial" description="等待音频输入..." size="small" style="margin:16px 0;"></n-empty>
            <div v-for="line in finals" :key="line.key" class="transcript-line">
              <span class="t" v-if="line.start != null">[{{ fmtMs(line.start) }}]</span>{{ line.text }}<n-text v-if="line.words" depth="3" style="font-size:0.8em;"> ({{ line.words }} 词)</n-text>
            </div>
          </div>
          <n-text v-if="partial" italic depth="2" style="display:block;margin-top:6px;">{{ partial }}</n-text>
        </n-card>

        <n-card size="small">
          <n-space justify="space-between" align="center">
            <n-button text style="font-size:1em;font-weight:600;" @click="logOpen = !logOpen">🧾 协议日志 {{ logOpen ? '▲' : '▼' }}</n-button>
            <n-button v-if="logOpen" size="tiny" tertiary @click="logs.length = 0">清空</n-button>
          </n-space>
          <div v-if="logOpen" ref="logRef" class="proto-log" style="margin-top:10px;">
            <div v-for="l in logs" :key="l.key" :class="l.kind">{{ l.ts }} {{ l.kind === 'send' ? '→' : l.kind === 'recv' ? '←' : '•' }} {{ l.text }}</div>
            <div v-if="!logs.length">（暂无消息）</div>
          </div>
        </n-card>
      </div>`,
  };

  createApp(makeRoot(AppBody)).use(naive).mount('#app');
})();
